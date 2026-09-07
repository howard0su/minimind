import time
import argparse
import json
import math
import random
import warnings
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def iter_jsonl(files):
    for file_path in files:
        with file_path.open('r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)


def prepare_ppl_sample(sample, tokenizer, is_pretrain, max_length):
    if is_pretrain:
        tokens = tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=max_length - 2,
            truncation=True
        ).input_ids
        input_ids = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        labels = input_ids.copy()
    else:
        messages = []
        tools = None
        for message in sample['conversations']:
            message = dict(message)
            if message.get('role') == 'system' and message.get('tools'):
                tools = json.loads(message['tools']) if isinstance(message['tools'], str) else message['tools']
            if message.get('tool_calls') and isinstance(message['tool_calls'], str):
                message['tool_calls'] = json.loads(message['tool_calls'])
            messages.append(message)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
        input_ids = tokenizer(prompt).input_ids[:max_length]
        labels = [-100] * len(input_ids)
        assistant_bos = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        assistant_eos = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        index = 0
        while index < len(input_ids):
            if input_ids[index:index + len(assistant_bos)] != assistant_bos:
                index += 1
                continue
            start = index + len(assistant_bos)
            end = start
            while end < len(input_ids) and input_ids[end:end + len(assistant_eos)] != assistant_eos:
                end += 1
            for label_index in range(start, min(end + len(assistant_eos), len(input_ids))):
                labels[label_index] = input_ids[label_index]
            index = end + len(assistant_eos)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


@torch.inference_mode()
def calculate_dataset_perplexity(model, tokenizer, args):
    if args.ppl_samples < 1:
        raise ValueError('--ppl_samples must be at least 1.')
    if args.ppl_max_length < 0:
        raise ValueError('--ppl_max_length cannot be negative.')

    is_pretrain = 'pretrain' in args.weight
    pattern = 'pretrain_*.jsonl' if is_pretrain else 'sft_*.jsonl'
    dataset_files = sorted(Path(args.dataset_dir).glob(pattern))
    if not dataset_files:
        raise FileNotFoundError(f'No dataset files found: {Path(args.dataset_dir) / pattern}')

    max_length = args.ppl_max_length or (340 if is_pretrain else 768)
    model_max_length = getattr(model.config, 'max_position_embeddings', None) or max_length
    max_length = min(max_length, model_max_length)
    if max_length < 2:
        raise ValueError('PPL maximum length must be at least 2.')
    total_nll = 0.0
    total_tokens = 0
    sample_count = 0

    for sample in iter_jsonl(dataset_files):
        input_ids, labels = prepare_ppl_sample(sample, tokenizer, is_pretrain, max_length)
        target_tokens = int((labels[1:] != -100).sum().item())
        if target_tokens == 0:
            continue

        input_ids = input_ids.unsqueeze(0).to(args.device)
        labels = labels.unsqueeze(0).to(args.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=labels
        )
        total_nll += outputs.loss.float().item() * target_tokens
        total_tokens += target_tokens
        sample_count += 1
        if sample_count >= args.ppl_samples:
            break

    if sample_count == 0:
        raise ValueError('No valid samples with target tokens were found.')

    mean_loss = total_nll / total_tokens
    return math.exp(mean_loss), mean_loss, total_tokens, sample_count, dataset_files


def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--dataset_dir', default='dataset', type=str, help="PPL评测数据目录")
    parser.add_argument('--ppl_samples', default=20, type=int, help="PPL快速评测使用的样本数")
    parser.add_argument('--ppl_max_length', default=0, type=int, help="PPL单样本最大长度（0=按训练配置自动选择）")
    parser.add_argument('--device', default='cpu' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n[2] PPL计算\n'))
    if input_mode == 2:
        ppl, loss, token_count, sample_count, dataset_files = calculate_dataset_perplexity(model, tokenizer, args)
        print(f'[Dataset]: {", ".join(str(path) for path in dataset_files)}')
        print(f'[Samples]: {sample_count} | [Tokens]: {token_count}')
        print(f'[PPL]: {ppl:.4f} | [Loss]: {loss:.4f}\n')
        return

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()