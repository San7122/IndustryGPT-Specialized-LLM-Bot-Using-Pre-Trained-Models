# EduBot: Industry-Specific LLM for Education and Training

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Model](https://img.shields.io/badge/model-Llama--2--7B--Chat-green)
![Fine-tuning Method](https://img.shields.io/badge/method-QLoRA-orange)

## 📚 Overview

**EduBot** is an industry-specific Large Language Model (LLM) chatbot designed to assist education and training professionals with pedagogical theories, instructional design, educational technology, assessment strategies, and professional development. Built by fine-tuning the **Llama-2-7B-Chat** model using **QLoRA (Quantized Low-Rank Adaptation)** on curated education-domain data.

This project demonstrates that meaningful domain adaptation of large language models is feasible even with constrained computational resources and modest dataset sizes, making it accessible to educators and researchers without enterprise-grade computing infrastructure.

## 🎯 Key Features

- ✅ **Domain-Specific Knowledge**: Specialized responses for education/training queries
- ✅ **Parameter-Efficient**: QLoRA fine-tuning requires only 16GB GPU VRAM
- ✅ **Fast Training**: 20 epochs complete in ~45 minutes on T4 GPU
- ✅ **Reproducible**: All code and configuration included
- ✅ **Open-Source**: Built on Llama-2, freely available for research
- ✅ **Covers 7 Education Sub-Domains**: Pedagogical theories, instructional design, EdTech, assessment, classroom management, professional development, and policy/equity

## 📊 Dataset & Training

### Dataset Composition
- **35 curated instruction-response pairs** covering:
  - Pedagogical theories and frameworks (Bloom's Taxonomy, Constructivism)
  - Instructional design methodologies (ADDIE, Backward Design)
  - Educational technology integration (LMS, adaptive learning, AI)
  - Assessment and evaluation strategies
  - Classroom management approaches
  - Professional development topics
  - Policy and equity issues

- **Format**: Llama-2-Chat prompt template with system prompts
- **Response Length**: 150-300 words per example
- **Max Tokens**: 512 for input sequences

### Training Configuration

| Hyperparameter | Value |
|---|---|
| **Base Model** | Llama-2-7B-Chat |
| **Fine-tuning Method** | QLoRA |
| **Quantization** | 4-bit NormalFloat |
| **LoRA Rank** | 64 |
| **LoRA Alpha** | 16 |
| **Learning Rate** | 2e-4 |
| **Batch Size** | 4 |
| **Epochs** | 20 |
| **Optimizer** | Paged AdamW (32-bit) |
| **GPU** | NVIDIA Tesla T4 (16GB VRAM) |
| **Training Time** | ~45 minutes |
| **Memory Required** | 3.5 GB (quantized) |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA 11.8+ (for GPU support)
16GB GPU VRAM (T4 GPU or better)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/EduBot.git
cd EduBot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.7.0
bitsandbytes>=0.41.0
datasets>=2.14.0
accelerate>=0.24.0
trl>=0.7.0
pydantic>=2.0.0
```

## 📖 Usage

### 1. Load the Fine-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Load the fine-tuned model
model_name = "path/to/edubot-model"
model = AutoPeftModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Merge LoRA weights with base model (optional)
merged_model = model.merge_and_unload()
```

### 2. Generate Responses

```python
def generate_response(question: str, max_length: int = 512):
    """Generate EduBot response to an education-related question."""
    
    # Format prompt
    prompt = f"""[INST] <<SYS>>
You are EduBot, an expert AI assistant specialized in the Education and Training industry.
Provide comprehensive, evidence-based responses that reference established educational frameworks.
<</SYS>>

{question} [/INST]"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
question = "What is Bloom's Taxonomy and how can I use it in my curriculum design?"
response = generate_response(question)
print(response)
```

### 3. Run Interactive Chat

```bash
python chat.py
```

```
EduBot: Welcome! I'm EduBot, your education and training assistant.
        Ask me about pedagogical theories, instructional design, 
        educational technology, assessment strategies, and more!

You: What is the difference between formative and summative assessment?

EduBot: [Detailed, comprehensive response based on educational theory]

You: quit
```

## 📁 Project Structure

```
EduBot/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── data/
│   └── training_data.json            # 35 education instruction-response pairs
│
├── notebooks/
│   ├── 01_data_preparation.ipynb     # Data curation and formatting
│   ├── 02_fine_tuning.ipynb          # QLoRA fine-tuning on Google Colab
│   ├── 03_evaluation.ipynb           # Model evaluation and comparison
│   └── 04_inference.ipynb            # Running inference examples
│
├── src/
│   ├── data_loader.py                # Dataset loading utilities
│   ├── model_utils.py                # Model loading and inference
│   ├── training.py                   # Fine-tuning script
│   └── chat.py                       # Interactive chatbot interface
│
├── models/
│   └── edubot-llama2-7b-qlora/      # Fine-tuned model (LoRA weights)
│       ├── adapter_config.json
│       ├── adapter_model.bin
│       └── ...
│
├── results/
│   ├── training_loss.csv             # Training metrics
│   ├── qualitative_evaluation.txt    # Q&A examples
│   └── comparison_base_model.txt     # vs Llama-2-Chat baseline
│
├── docs/
│   ├── methodology.md                # Detailed methodology
│   ├── limitations.md                # Known limitations
│   └── future_work.md                # Improvement roadmap
│
└── research_paper/
    └── EduBot_Research_Paper_Full_12pages.docx
```

## 🔧 Fine-Tuning Your Own Model

If you want to reproduce the fine-tuning process:

```bash
# 1. Prepare your data (format as JSON with "instruction" and "response" fields)
python src/data_loader.py --input data/training_data.json --validate

# 2. Run fine-tuning on Google Colab
# Upload notebooks/02_fine_tuning.ipynb to Colab with GPU enabled
# Follow the instructions in the notebook

# 3. Evaluate the model
python notebooks/03_evaluation.ipynb

# 4. Save the fine-tuned model
# LoRA weights are automatically saved to ./models/edubot-llama2-7b-qlora/
```

## 📊 Results & Performance

### Training Metrics
- **Convergence**: Stable convergence across all 20 epochs
- **Final Training Loss**: Significantly reduced from initial values
- **Evaluation Strategy**: Epoch-level evaluation on validation set
- **No Catastrophic Forgetting**: Maintained base model capabilities

### Qualitative Evaluation

#### Test on Training Data ✅
- Successfully reproduced domain-specific content
- Maintained structured response format (definition → explanation → application → considerations)
- Correctly used educational terminology

#### Generalization to Novel Questions ✅
- Model produced relevant responses to out-of-distribution questions
- Drew on fine-tuned domain knowledge + base model pre-training
- Appropriate contextual reasoning for education topics

#### Comparison with Base Model 📈

| Aspect | Base Llama-2-Chat | Fine-tuned EduBot |
|---|---|---|
| Response Length | Short (50-150 words) | Comprehensive (150-300 words) |
| Domain Terminology | Generic education references | Precise pedagogical terminology |
| Framework References | Rare/generic | Frequent and specific (Bloom's, ADDIE, etc.) |
| Practical Guidance | Limited | Detailed with examples |
| **Overall Quality** | **General-purpose** | **Domain-expert level** |

## 🎓 Example Interactions

### Example 1: Pedagogical Frameworks
```
Q: What is Bloom's Taxonomy and how can I use it in my curriculum design?

EduBot: Bloom's Taxonomy is a hierarchical framework for categorizing 
educational learning objectives, originally developed by Benjamin Bloom in 1956 
and revised by Anderson and Krathwohl in 2001. The revised taxonomy consists 
of six cognitive levels:

1. Remember: Recall facts and basic concepts
2. Understand: Explain ideas or concepts
3. Apply: Use information in new situations
4. Analyze: Draw connections between ideas
5. Evaluate: Justify decisions and choices
6. Create: Produce new or original work

In curriculum design, you can use Bloom's Taxonomy to:
- Define learning objectives at different cognitive levels
- Design assessments that evaluate multiple levels of thinking
- Scaffold instruction from lower to higher-order thinking skills
- Ensure balanced assessment across cognitive domains
[... continues with practical examples ...]
```

### Example 2: Instructional Design
```
Q: Explain the ADDIE model for instructional design.

EduBot: ADDIE is a systematic instructional design model consisting of 
five phases: Analysis, Design, Development, Implementation, and Evaluation...
[Domain-specific comprehensive response with practical applications]
```

## ⚠️ Limitations & Future Work

### Current Limitations
1. **Limited Dataset**: Only 35 training examples constrains knowledge breadth
2. **Hallucination Risk**: Model may generate plausible but inaccurate information
3. **Token Limitations**: 512-token max may truncate long-form responses
4. **Qualitative Evaluation Only**: Lacks systematic quantitative metrics
5. **Domain Coverage**: Non-uniform coverage of education sub-topics

### Future Improvements
- 🔄 **Retrieval-Augmented Generation (RAG)**: Ground responses in authoritative documentation
- 📚 **Expanded Dataset**: Scale to 500-1000+ high-quality examples
- 📊 **Quantitative Metrics**: Implement BLEU, ROUGE, BERTScore evaluation
- 👥 **Expert Evaluation**: Panel of education professionals for quality assessment
- 🌐 **Web Deployment**: Streamlit/Gradio interface for real-world testing
- 💬 **Multi-turn Conversations**: Context-aware follow-up responses
- 🛡️ **Safety Guardrails**: Pedagogically sound advice validation

## 🔬 Research Contributions

This project demonstrates:
- ✅ Feasibility of domain-specific LLM adaptation with limited resources
- ✅ QLoRA effectiveness for education domain fine-tuning
- ✅ Meaningful improvements possible with <50 training examples
- ✅ Reproducible methodology for education sector AI tools
- ✅ Value of parameter-efficient fine-tuning for accessibility

## 📄 Research Paper

Full academic paper available in `research_paper/EduBot_Research_Paper_Full_12pages.docx`

**Key Sections:**
- Literature Review on LLMs in Education & Parameter-Efficient Fine-Tuning
- Complete Methodology with Data Curation Details
- Results & Qualitative Evaluation Analysis
- Practical Applications in Education Industry
- Future Work and Conclusions

**Citation:**
```bibtex
@article{thakur2026edubot,
  title={Developing an Industry-Specific Large Language Model Bot for the Education and Training Sector: Fine-Tuning Llama-2 Using QLoRA on Domain-Specific Data},
  author={Thakur, Sanjana},
  year={2026},
  note={Industry Immersion Module Research Paper, Woolf University}
}
```

## 🛠️ Contributing

Contributions are welcome! Please feel free to:
- Report issues and bugs
- Suggest improvements
- Submit pull requests
- Share education domain datasets
- Propose new features

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Important**: The base Llama-2 model is subject to Meta's Acceptable Use Policy. Please review it before using this model.

## 🤝 Acknowledgments

- **Meta AI** for Llama-2 model release
- **Hugging Face** for transformers library and model hosting
- **Peft Team** for QLoRA implementation
- **Google Colab** for free GPU resources
- Educational frameworks and pedagogical research cited throughout

## 📞 Contact & Support

- **Author**: Sanjana Thakur
- **Student ID**: 4295000008
- **Email**: [your.email@example.com]
- **Institution**: Woolf University

For questions or collaboration opportunities, please open an issue or reach out directly.

## 🗂️ Related Resources

- [Llama-2 Research Paper](https://arxiv.org/abs/2307.09288)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [PEFT Library](https://github.com/huggingface/peft)

## 🔗 Model Card

**Model Name**: EduBot-Llama-2-7B-QLoRA  
**Base Model**: Llama-2-7B-Chat (Meta AI)  
**Fine-tuning Method**: QLoRA  
**Training Data**: 35 education-domain instruction-response pairs  
**Language**: English  
**Use Case**: Education and training domain assistance  
**Training Date**: 2026  
**Recommended Use**: Educational support, curriculum planning, instructional design guidance  
**Not Recommended For**: Critical legal/medical decisions, real-time education policy making without expert review

---

**Made with ❤️ for the education community**

⭐ If you find this project useful, please consider giving it a star!
