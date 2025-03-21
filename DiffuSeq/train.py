"""
Train a diffusion model on images.
"""
import sys
sys.path.append("/home/kara/.local/lib/python3.8/site-packages/")
import argparse
import json, torch, os
import torch.nn as nn
import numpy as np
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed, BertModel
import wandb

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

class ToxicityClassifier(nn.Module):
    def __init__(self):
        super(ToxicityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    data = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb=model_weight # use model's weights as init
    )
    next(data)

    data_valid = load_data_text(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight # using the same embedding wight with tranining data
    )

    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    args.device = dist_util.dev()
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    # print('#'*30, 'cuda', dist_util.dev())
    model.to(dist_util.dev()) #  DEBUG **
    # model.cuda() #  DEBUG **


# pre-trained model !!!!!
    pre_path = "/home/kara/DiffuSeq/diffusion_models/diffuseq_qqp_h128_lr0.0001_t2000_sqrt_lossaware_seed102_learned_mask_fp16_denoise_0.5_reproduce20240610-07:44:13/ema_0.9999_040000.pt"
    model.load_state_dict(torch.load(pre_path),strict=False)
# 修改encoder 
    # 初始化预训练模型
    pretrained_encoder = ToxicityClassifier()

    # 加载预训练的编码器权重
    pretrained_state_dict = torch.load('/home/kara/classification/model/7.21-2layer/model_epoch_14.pth', map_location=args.device)
    # 获取预训练模型和新模型的状态字典
    pretrained_state_dict = pretrained_encoder.state_dict()
    new_model_state_dict = model.input_transformers.state_dict()

    # 只加载匹配的权重
    matched_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in new_model_state_dict and v.size() == new_model_state_dict[k].size()}
    new_model_state_dict.update(matched_state_dict)

    # 加载更新后的状态字典到新模型
    model.input_transformers.load_state_dict(new_model_state_dict)
    print(model)
    # # 冻结
    # for layer in model.input_transformers.layer[:11]:
    #     for param in layer.parameters():
    #         param.requires_grad = False


# -------------
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    args.device = ""
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project="diffuseq",
            name=args.checkpoint_path,
            notes="2nd-run"
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval
    ).run_loop()

if __name__ == "__main__":
    main()
