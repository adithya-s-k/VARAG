# 👁️👁️ VARAG (Vision Augmented Retrieval and Generation)

| ![VARAG](./docs/assets/llama.png)| VARAG (Vision-Augmented Retrieval and Generation) is a vision-first RAG engine that emphasizes vision-based retrieval techniques. It enhances traditional Retrieval-Augmented Generation (RAG) systems by integrating both visual and textual data through Vision-Language models. |
|:--:|:--|

## Supported Retrieval Techniques

VARAG supports multiple retrieval techniques:

- Colpali
- Vision Encoder Based:
  - Seglip
  - CLIP
  - Jina CLIP
- OCR-based Text RAG

## 🚀 Getting Started with VARAG

Follow these steps to set up VARAG:

### 1. Clone the Repository

```bash
git clone https://github.com/adithya-s-k/VARAG
cd VARAG
```

### 2. Set Up Environment

Create and activate a virtual environment using Conda:

```bash
conda create -n varag-venv python=3.10
conda activate varag-venv
```

### 3. Install Dependencies

Install the required packages using pip:

```bash
pip install -e .
```

## Running VARAG

### Demo

To run the demo:

```bash
python demo.py
```

### Server

To start the VARAG server:

```bash
python server.py
```

## API Endpoints

VARAG provides the following API endpoints:

- Configure
- Ingest
- Search
- Query

## Docker

Docker support is available for VARAG. (You may want to add specific instructions for using Docker here.)

## Explaining Vision Diagrams

VARAG has the capability to explain vision diagrams. (You might want to elaborate on this feature and how it works.)

## 🛠️ Contributing

Contributions to VARAG are highly encouraged! Whether it's code improvements, bug fixes, or feature enhancements, feel free to contribute to the project repository. Please adhere to the contribution guidelines outlined in the repository for smooth collaboration.

## 📜 License

VARAG is licensed under the [MIT License](https://opensource.org/licenses/MIT), granting you the freedom to use, modify, and distribute the code in accordance with the terms of the license.

## 🙏 Acknowledgments

We extend our sincere appreciation to the following projects and their developers:

- Docling - For PDF text extraction (OCR)
- LanceDB - For Vector Database functionality
- Developers of Surya, Marker, GPT-4 Vision, and various other tools and libraries that have played pivotal roles in the success of this project.

Additionally, we are grateful for the support of the open-source community and the invaluable feedback from users during the development journey.
