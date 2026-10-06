# Attribution

This project builds on the open models, datasets, voices and code below. Licences are as stated on each source's
model card or repository on 2026-10-07. Check the source for the current terms.

## Models

| What | Used for | Source | Licence |
|---|---|---|---|
| PersonaPlex 7B (NVIDIA) | Base of the speech agent; our Hinglish LoRA is trained on it | [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1), code [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) | NVIDIA Open Model License (weights), MIT (code) |
| Moshi / Mimi (Kyutai) | PersonaPlex's backbone and audio codec | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi), [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16) | Apache-2.0 (code) |
| moshi-finetune (Kyutai) | LoRA training code (we ship a patch against it) | [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) | Apache-2.0 |
| Needle 3 (Cactus Compute) | Base of the action router; our router is a LoRA fine-tune of it | [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3) | Apache-2.0 |
| Gemma 4 31B-it (Google) | Generated the scenarios, records and dialogues; LLM judge in the evaluation | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | Apache-2.0 (per model card) |
| Whisper-Hinglish (Trelis) | ASR: audio QC of the dataset, router transcripts, evaluation | [Trelis/whisper-hinglish-preview](https://huggingface.co/Trelis/whisper-hinglish-preview) (based on [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) and [ARTPARK-IISc/whisper-large-v3-vaani-hindi](https://huggingface.co/ARTPARK-IISc/whisper-large-v3-vaani-hindi)) | Apache-2.0 |
| IndicF5 Hindi-English code-switch TTS (Tharshan) | TTS for the V3 / V4 training data | [Tharshan/indicf5_hindi-english_code_switch](https://huggingface.co/Tharshan/indicf5_hindi-english_code_switch) (based on [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5)) | Apache-2.0 |
| Kokoro-82M (hexgrad) | TTS for the V1 data and the English control set | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache-2.0 |
| MMS forced aligner (torchaudio) | Word-level alignment of the synthetic calls | [torchaudio.pipelines.MMS_FA](https://pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html), [pytorch/audio](https://github.com/pytorch/audio) | BSD-2-Clause (code) |
| vLLM | Serving Gemma during data generation | [vllm-project/vllm](https://github.com/vllm-project/vllm) | Apache-2.0 |

## Voices

The IndicF5 TTS clones a voice from a short reference clip. These are the four reference clips used for the V3 / V4 data:

| Voice | Role in the data | Source | Licence / note |
|---|---|---|---|
| `ritu_hinglish` | Female agent | From the `voices/` folder of [Tharshan/indicf5_hindi-english_code_switch](https://huggingface.co/Tharshan/indicf5_hindi-english_code_switch/tree/main/voices). That repo states the clip was generated with Sarvam AI's Bulbul v3 ([sarvam.ai](https://www.sarvam.ai)). | Repo: Apache-2.0. Sarvam's terms for Bulbul output apply to the original clip. |
| `orato_male` | Male agent | From the `voices/` folder of [tryorato/orato-tts-hindi-v1](https://huggingface.co/tryorato/orato-tts-hindi-v1/tree/main/voices) | MIT |
| `orato_female` | Female customer | From the `voices/` folder of [tryorato/orato-tts-hindi-v1](https://huggingface.co/tryorato/orato-tts-hindi-v1/tree/main/voices) | MIT |
| `fleurs_hi_m1559` | Male customer | [google/fleurs](https://huggingface.co/datasets/google/fleurs), config `hi_in`, validation split, utterance id 1559: a recording of a real speaker | CC-BY-4.0. The male customer voice in the V3 / V4 data is a clone of this speaker. |

FLEURS citation:

> Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, Ankur Bapna.
> *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech.* 2022.
> [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)

The V1 data and the English control set use Kokoro's built-in voices: `hf_beta`, `hm_omega`, `hf_alpha`, `hm_psi` and the English voices.

## Notes

- Everything in the training data is synthetic: scenarios, customer records, names, order IDs, phone numbers and email addresses. The company/brand names of the agents are fictional. Restaurant, product and place names may be real, and some details look real, such as real email providers and valid-format phone numbers.
- The fine-tuned weights are derivatives of the models above and keep their base models' licences. See each model repo's card.
