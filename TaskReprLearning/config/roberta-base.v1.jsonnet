// Three token types (OTHER, TASK, LIST)
// Intent extractor (sum multi-head (12) attention w/ tanh activation)
// Type embeddings fed to the intent extractor
// Loss:
// Autocompletion
// COMET (xNedd, xIntent)
// FrameNet
{
  "dataset_reader": [
    {
      "type": "task-specification",
      "split": "train",
      "filepath": "data/v1/trn.json",
      "add_bos_to_input": true,
      "add_eos_to_input": true,
      "use_sep_as_eos": true,
      "lowercase": true,
      "framenet_label_path": "data/v1/fe_embs.roberta-base.768d.txt"
    },
    {
      "type": "task-specification",
      "split": "validation",
      "filepath": "data/v1/vld.json",
      "add_bos_to_input": true,
      "add_eos_to_input": true,
      "use_sep_as_eos": true,
      "lowercase": true,
      "framenet_label_path": "data/v1/fe_embs.roberta-base.768d.txt"
    },
  ],
  "model": {
    "type": "simple_encoder",
    "transformer": {
      "type": "roberta-base",
    },
    "input_type_embeddings": false,
    "primary_output": "discrete",
    "primary_loss_type": "ce",  // cross entropy
    "primary_loss_smoothing_factor": 0.1,
    "intent_extractor": {
      "num_layers": 0,
      "type_embedding_dim": 768,
      "initialization": "normal",
      "initialization_std": 0.02,
      "attention": {
        "type": "sum",
        "activation": "tanh",
        "num_attention_heads": 12,
        "dropout": 0.1,
      }
    },
    "decoder": {
      "type": "gru",
      "num_layers": 2,
      "hidden_dims": 768,
      "activations": "gelu",
      "dropout": 0.1,
      "bias": true,
      "cross_attention": true,
    },
    "auxiliary": [
      {
        "name": "comet-xNeed",
        "output_type": "generation",
        "type": "gru",
        "num_layers": 2,
        "hidden_dims": 768,
        "activations": "gelu",
        "bias": true,
        "dropout": 0.1,
      },
      {
        "name": "comet-xIntent",
        "output_type": "generation",
        "type": "gru",
        "num_layers": 2,
        "hidden_dims": 768,
        "activations": "gelu",
        "bias": true,
        "dropout": 0.1,
      },
      {
        "name": "framenet",
        "type": "gile",
        "activations": "gelu",
        "dropout": 0.1,
        "bias": true,
        "vocab_size": 459,
        "label_embedding_dim": 768,
      "framenet_label_path": "data/v1/fe_embs.roberta-base.768d.txt",
        "freeze_label_embedding": true,
        "joint_dim": 768,
        "output_type": "multilabel",
      }
    ]
  },
  "trainer": {
    "mtl_weighting_method": "muppet",
    "mtl_coef": 1.0,
    "batch_size": 100,
    "num_gradient_accumulation_steps": 24,
    "pretraining_num_epochs": 5,
    "pretraining_optimizer": {
      "type": "AdamW",
      "lr": 1e-3,
      "eps": 1e-6,
      "max_grad_norm": 1.0,
      "betas": [0.9, 0.999],
    },
    "pretraining_scheduler": {
      "type": "constant",
      "warmup_steps_ratio": 0.002
    },
    "label_weight": [
      {
        "name": "framenet",
        "label_weight_file_path": "data/framenet/fe_weights.v1.json",
      }
    ],
    "num_epochs": 10,
    "patience": 3,
    "optimizer": {
      "type": "AdamW",
      "lr": 1e-5,
      "eps": 1e-6,
      "max_grad_norm": 1.0,
      "betas": [0.9, 0.999],
    },
    "scheduler": {
      "type": "linear",
      "warmup_steps_ratio": 0.002
    }
  }
}
