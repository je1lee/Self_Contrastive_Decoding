import torch
from argparse import ArgumentParser


from transformers import AutoTokenizer
from modeling_opt import OPTForCausalLM

def run(args, prompt):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = OPTForCausalLM.from_pretrained(args.model_path,
    torch_dtype=torch.bfloat16,
    return_dict=True,
    device_map="auto",
    use_cache=True
    )

    prompt = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    model.outside = False
    
    model.contrastive =True
    generated_ids_contrastive = model.generate(prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True,
                                    output_hidden_states=True)
    
    model.contrastive = False
    generated_ids_argmax = model.generate(prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True,
                                    output_hidden_states=True)
    

    generated_text_contrastive = tokenizer.batch_decode(generated_ids_contrastive, skip_special_tokens=True)[0]
    generated_text_argmax = tokenizer.batch_decode(generated_ids_argmax, skip_special_tokens=True)[0]
    print("\nSelf Contrastive Output : \n", generated_text_contrastive)
    print("\nArgmax Output: \n", generated_text_argmax)

    return 


if __name__ == '__main__':
    prompt = 'Barrack Obama was born in Honolulu, Hawaii. He was born in'
    
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/ssd0/data/fast-llm/opt-13b/')
    parser.add_argument('--max_new_tokens', type=int, default=128)                                           
    args = parser.parse_args()
    run(args, prompt)