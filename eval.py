import ast
import json
import os

import torch

from argparse import ArgumentParser

import pandas as pd
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_opt import OPTForCausalLM

from typing import Mapping
import random

from SimCTG.simctg.evaluation import measure_repetition_and_diversity
from simcse import SimCSE

from datasets import load_dataset, DatasetDict

import mauve

simcse_model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
mauve_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def calc_metrics(ref, text, real_ref) -> Mapping:
    try:
        _, _, _, diversity = measure_repetition_and_diversity([text])
    except ZeroDivisionError:  # text is too short
        diversity = 0

    coherence = simcse_model.similarity(text, ref)
    mauve_score = calc_mauve_score(real_ref, text)
    print(mauve_score)
    return {'diversity': diversity,
            'coherence': coherence,
            'mauve': mauve_score}
    

def calc_mauve_score(ref, text):
    tgt_len = 32
    y = mauve_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=tgt_len)
    x = mauve_tokenizer.encode(ref, return_tensors="pt", truncation=True, max_length=len(y[0]))

    p_text = mauve_tokenizer.decode(x[0], skip_special_tokens=True)
    q_text = mauve_tokenizer.decode(y[0], skip_special_tokens=True)
    print("p_text : ", p_text)
    print("q_text : ", q_text)
    out = mauve.compute_mauve(p_text=p_text, q_text=q_text, max_text_length=len(x[0]), verbose=False, featurize_model_name="gpt2")
    return out.mauve
    
def get_texts_for_inference(dataset, num_samples, dataset_split="train"):
    HF_AUTH_TOKEN = "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"
    wikinews = load_dataset('bigscience-data/roots_en_wikinews', use_auth_token=HF_AUTH_TOKEN)
    wikinews = wikinews.map(wikinews_dataset_preprocess, batched=True, remove_columns=["text","meta"])
    dataset_dict = wikinews.rename_column("text", "source_text")
    texts = dataset_dict[dataset_split]['source_text']

    if num_samples is not None and num_samples < len(texts):
        random.seed(42)
        sample_ids = random.Random(0).sample(list(range(len(texts))), k=num_samples)
        texts = [texts[i] for i in sample_ids]

    return texts

def wikinews_dataset_preprocess(examples):
    def get_main_text(news_text):
        return sorted(news_text.split(":"), key=len)[-1].strip()

    def filter_wikinews(source_text):
        text = get_main_text(source_text)
        return len(text.split()) > 50

    def preprocess_wikinews(text):
        main_text = get_main_text(text)
        main_text = main_text.replace('Pillars of Wikinews writing Writing an article ', '')
        return main_text

    source_column = 'text'
    sources = []
    for source_text in examples[source_column]:
        if source_text is not None and filter_wikinews(source_text):
            sources.append(preprocess_wikinews(source_text))

    new_examples = {
        source_column: sources,
    }

    return new_examples

def run(args):
    texts = get_texts_for_inference(args.dataset, args.number_of_samples, args.dataset_split)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = OPTForCausalLM.from_pretrained(args.model_path,
    torch_dtype=torch.bfloat16,
    return_dict=True,
    device_map="auto",
    use_cache=True
    )
    if args.contrastive:
        model.contrastive =True
    else:
        model.contrastive = False

    if args.outside:
        student_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m",
                                                                device_map="auto",
                                                                torch_dtype=torch.bfloat16,
                                                                return_dict=True)
        model.student_model = student_model
        model.outside = True
        print("CD")
    else:
        model.outside = False
        print("SCD")
        pass

    info = 'top_k' if args.use_top_k else ('top_p' if args.use_top_p else f'beam_{args.beam_search}')
    desc = f'{args.model_path.replace("/", "_")}_{info}'

    results = []
    print(f'generating texts from {len(texts)} prompts')
    for i, text in tqdm(enumerate(texts), total=len(texts)):
        prompt_text = ' '.join(text.split()[:args.num_tokens])

        real_ref = ' '.join(text.split()[args.num_tokens:args.num_tokens+args.max_new_tokens])
        prompt = tokenizer(prompt_text, return_tensors='pt', truncation=True).input_ids.to(model.device)
        prompt = "Barrack Obama was born in Honolulu, Hawaii. He was born in"
        prompt = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        generated_ids = model.generate(prompt,
                                       max_new_tokens=args.max_new_tokens,
                                       num_beams=args.beam_search,
                                       do_sample=args.use_top_k or args.use_top_p,
                                       top_p=0.95 if args.use_top_p else 1.0,
                                       top_k=50 if args.use_top_k else 0.0,
                                       output_hidden_states=True)

        generated_ids = generated_ids[:, prompt.shape[1]:]

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)
        # print(generated_text)
        example_metrics = calc_metrics(prompt_text, generated_text, real_ref)
        results.append({**example_metrics})

    output_dir = os.path.join('output', desc)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(results)

    metrics = {'diversity': df['diversity'].mean(), 'coherence': df['coherence'].mean(), 'mauve': df["mauve"].mean(),
               'num_examples': len(df)}
    print(metrics)
    with open(os.path.join(output_dir, f'metrics_{args.dataset}.json'), 'w') as f:
        f.write(json.dumps(metrics))

    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/ssd0/data/fast-llm/opt-13b/')
    parser.add_argument('--contrastive', type=ast.literal_eval, default=False)
    parser.add_argument('--outside', type=ast.literal_eval, default=False)
    parser.add_argument('--dataset', type=str, required=True, default="wikinews")
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--num_tokens', type=int, default=32)                                               
    parser.add_argument('--beam_search', type=int, default=1)
    parser.add_argument('--use_top_k', type=ast.literal_eval, default=False)
    parser.add_argument('--use_top_p', type=ast.literal_eval, default=False)
    parser.add_argument('--dataset_split', type=str, default='train')
    parser.add_argument('--number_of_samples', type=int, default=None)
    parser.add_argument('--output_dir', '-o', type=str, required=False)
    args = parser.parse_args()
    run(args)