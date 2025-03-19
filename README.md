# MINILLM – Simple Chatbot Based on GPT

**MINILLM** is a compact, self-developed LLM (Large Language Model). It aims to provide a better understanding of AI models for both myself and others. It offers a simple way to train, test, and customize a language model. This project is intended for both enthusiasts and developers who want to experiment on their own. Anyone can copy this project to their computer.

## 1. Overview of MINILLM

### Comparison of Popular LLMs

| Model       | Developer      | Key Features                                      |
|------------|--------------|--------------------------------------------------|
| **ChatGPT**  | OpenAI        | Versatile chatbot AI, trained on vast datasets |
| **Grok**     | xAI (Elon Musk) | Humorous chatbot with sharp responses          |
| **DeepSeek** | Open-Source   | Efficient and free LLM from China              |

MINILLM is a **minimalist version of these models**, running locally and fully customizable.

## 2. The Complete Process – From Model to Chatbot

The development of a language model like MINILLM consists of several steps:

1. **Load a pre-trained model** – We use an already trained model (GPT-2) from Hugging Face instead of training one from scratch.
2. **Provide datasets for training** – The model is further trained with a dataset (e.g., OpenWebText) to adapt it to desired inputs.
3. **Tokenization & Processing** – Text is converted into tokens that the model can understand. Short texts are padded, and long texts are truncated if necessary.
4. **Training the Model** – The model runs through the training data multiple times (epochs) and learns to predict text by calculating the next token based on previous tokens. Weights in the neural network are adjusted, enabling the model to recognize patterns and improve.
5. **Fine-tuning and Testing** – After training, the model is tested and can be used for various applications.

### What is OpenWebText as a Dataset?

OpenWebText is a freely available dataset consisting of high-quality web texts. It was created as an alternative to OpenAI’s GPT-2 training data and contains texts from reputable sources filtered based on specific quality criteria.

### Why Must All Inputs Have the Same Length? (Padding & Truncation)

Neural networks operate more efficiently when all inputs have the same length. To ensure the model trains in fixed blocks:
- **Short inputs are padded with a `[PAD]` or EOS token**
- **Long inputs are truncated to the maximum length**

Without padding, batch processing and parallel training would be inefficient since the model would need to perform individual calculations for each input.

### What is a Neural Network and What Are Weights? (Simplified Explanation)

- A **neural network** can be imagined as a vast network of connections processing data. Each **neuron** acts like a small switch deciding which information is passed on.
- **Weights** are numerical values determining the strength of a connection between neurons. When the model is trained, it adjusts these weights to make better decisions.
- Initially, the model makes many mistakes, but through repeated training, it learns which words typically follow others.

### How Does the Model Predict the Next Text?

The model analyzes a sequence of words and calculates which word is most likely to follow next. This process happens in several steps:
1. **Convert input text into tokens** – The text is transformed into numerical values.
2. **A neural network processes the tokens** – The sequence of tokens is processed through multiple network layers.
3. **Calculate probabilities for the next token** – The model generates a list of possible next tokens with respective probabilities.
4. **Select the most likely token** – The model chooses the most probable word and outputs it as the next token.
5. **Repeat until the maximum text length is reached** – This process continues until the desired text length is generated.

## 3. What is Hugging Face and Why Do We Use It?

**Hugging Face** is a platform for open-source AI models. It provides pre-trained models, datasets, and libraries that simplify working with LLMs. In this project, we use:

- **Transformers**: A library for loading and training pre-trained models.
- **Datasets**: Enables loading and preparing text datasets.
- **Hugging Face Hub**: A platform to upload and share models with others.

This project uses GPT-2 from Hugging Face, but other models can also be used. Alternatives include:
- **GPT-2 Medium/Large/XL** – Larger versions of GPT-2 for improved performance.
- **GPT-Neo (EleutherAI)** – An open-source model with more advanced architectures.
- **BLOOM** – A multilingual open-source LLM.

## 4. Features and Limitations

### Features

- **Text Generation**: The model generates text based on user input.
- **Customizability**: Parameters such as training duration, token length, or batch size can be adjusted.
- **Offline Usage**: No API access required, as the model runs locally.
- **Custom Training Data**: Any dataset can be used for fine-tuning.

### Limitations

- **No long-term memory**: Limited ability to remember previous interactions.
- **Limited training data**: The model is trained on only about 10,000 OpenWebText samples (adjustable parameter).
- **Simple architecture**: No advanced training methods like RLHF (Reinforcement Learning from Human Feedback).

## 5. Installation and Usage

### Clone the Project and Set Up the Environment

```bash
git clone https://github.com/your-username/MINILLM.git
cd MINILLM
python3 -m venv venv
```
Depending on your operating system, activate the virtual environment:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

### Install Dependencies

The project requires some libraries that can be installed via `requirements.txt`:
```bash
pip install -r requirements.txt
```
This includes:
- **transformers**: Library from Hugging Face for working with pre-trained models.
- **torch**: PyTorch for training neural networks.
- **datasets**: A helper library for loading and processing datasets.
- **accelerate**: Optimizes training on different hardware setups.
- **huggingface_hub**: Interface for interacting with the Hugging Face platform.

### Train the Model (Takes 1–6 Hours Depending on Hardware)

```bash
python train_llm.py
```
In `train_llm.py`, the dataset can be changed by modifying the following line:
```python
dataset = load_dataset("openwebtext", trust_remote_code=True)
```
Alternatively, you can use your own training data, such as:
- **Wikipedia data** (`wikipedia` via Hugging Face Datasets)
- **Common Crawl** (Web text data)
- **Personal documents or specialized texts**

## 6. Adjustable Parameters

| Parameter                     | Description                                      | Default Value | Alternative Values |
|------------------------------|------------------------------------------------|--------------|----------------|
| `num_train_epochs`            | Number of training epochs                        | `2`          | `1 - 10` (depends on performance) |
| `max_length`                  | Maximum token length for inputs                 | `256`        | `128, 512, 1024` |
| `per_device_train_batch_size` | Number of samples per training step             | `2`          | `1 - 16` (depends on RAM) |
| `learning_rate`               | Learning rate of the model                      | `5e-5`       | `1e-4, 3e-5` |
| `tokenizer.pad_token`         | Default pad token for GPT-2                     | `tokenizer.eos_token` | `"[PAD]"` |

## 7. Key Terms Explained in Simple Words

- **Token**: The smallest unit of a text (word or character) that the model processes.
- **Tokenization**: The process of converting text into individual tokens before the model can understand it.
- **Batched** Processing: Training and inference are done in batches (groups of data) to optimize computational efficiency.
- **Truncation**: Cutting off long inputs to fit within the model’s maximum token limit.
- **Padding**: Filling shorter inputs with extra tokens to ensure all inputs have the same length.
- **Epochs**: The number of times the model goes through the entire dataset during training.
- **Pipeline**: A simplified Hugging Face interface for handling various NLP tasks.
- **Dataset**: A collection of texts or data used for training the model.

## 8. Interesting Facts About AI

- **10-15% of total computing power** in modern LLMs is used to reduce bias in generated text.
- **Modern AI models have more parameters than the human brain has synapses** – GPT-4 is estimated to have over **one trillion** parameters.
- **Data quality is more important than data quantity** – A small, clean dataset often yields better results than massive amounts of unstructured data.
- **AI requires constant fine-tuning** – Even advanced models must be retrained regularly to stay up to date.

## 9. Further Development

- More training data for better results.
- Creating a web interface with Gradio or Streamlit.
- Uploading the model to Hugging Face.

Feel free to contribute, provide feedback, or collaborate!



German/Deutsch 🇩🇪

# MINILLM – Simpler Chatbot auf GPT-Basis

**MINILLM** ist ein kompaktes, selbst entwickeltes LLM (Large Language Model). Es zielt darauf ab, sowohl mir als auch anderen ein besseres Verständnis für KI-Modelle zu vermitteln. Es bietet eine einfache Möglichkeit, ein Sprachmodell zu trainieren, zu testen und anzupassen. Dieses Projekt richtet sich sowohl an Interessierte als auch an Entwickler, die selbst experimentieren möchten. Dabei kann hier jeder das Projekt auf seinen Rechner kopieren.

## 1. Überblick über MINILLM

### Bekannte LLMs im Vergleich

| Modell       | Entwickler      | Besonderheit                                            |
| ------------ | --------------- | ------------------------------------------------------- |
| **ChatGPT**  | OpenAI          | Vielseitige Chat-KI, trainiert auf riesigen Datensätzen |
| **Grok**     | xAI (Elon Musk) | Humorvoller Chatbot mit zugespitzten Antworten          |
| **DeepSeek** | Open-Source     | Leistungseffizientes, freies LLM aus China              |

MINILLM ist eine **minimalistische Version dieser Modelle**, die lokal läuft und anpassbar ist.

## 2. Der gesamte Prozess – Vom Modell zum Chatbot

Der Aufbau eines Sprachmodells wie MINILLM verläuft in mehreren Schritten:

1. **Ein vortrainiertes Modell laden** – Wir nutzen ein bereits existierendes Modell (GPT-2) von Hugging Face, anstatt ein Modell von Grund auf zu trainieren. 
2. **Datensätze zum Trainieren bereitstellen** – Das Modell wird mit einem Datensatz (z. B. OpenWebText) weitertrainiert, um es besser auf gewünschte Eingaben anzupassen.
3. **Tokenization & Verarbeitung** – Texte werden in Tokens umgewandelt, damit das Modell sie verstehen kann. Kürzere Texte werden mit Padding ergänzt, lange Texte gegebenenfalls abgeschnitten.
4. **Training des Modells** – Das Modell durchläuft die Trainingsdaten mehrmals (Epochen) und lernt, Texte vorherzusagen, indem es das nächste Token basierend auf vorherigen Tokens berechnet. Dabei werden Gewichte in den neuronalen Netzwerken angepasst, sodass das Modell Muster erkennt und verbessert.
5. **Feinabstimmung und Tests** – Nach dem Training wird das Modell getestet und kann für verschiedene Anwendungen genutzt werden.

### Was ist OpenWebText für ein Datensatz?

OpenWebText ist ein frei zugänglicher Datensatz, der aus qualitativ hochwertigen Webtexten besteht. Er wurde als Alternative zu OpenAI’s GPT-2-Trainingsdaten erstellt und enthält Texte von vertrauenswürdigen Quellen, die nach bestimmten Qualitätskriterien gefiltert wurden.

### Warum müssen alle Eingaben die gleiche Länge haben? (Padding & Truncation)

Neuronale Netzwerke arbeiten effizienter, wenn alle Eingaben die gleiche Länge haben. Damit das Modell in festgelegten Blöcken trainieren kann, werden:
Kurze Eingaben mit einem Padding-Token ([PAD] oder EOS-Token) aufgefüllt
Lange Eingaben auf die maximale Länge gekürzt (Truncation)
Ohne Padding würden Batch-Verarbeitung und paralleles Training ineffizient laufen, weil das Modell sonst für jede Eingabe eine eigene Berechnung starten müsste.

### Was ist ein neuronales Netzwerk und was sind Gewichte? (Einfach erklärt)

Ein neuronales Netzwerk kann man sich wie ein riesiges Netz aus Verbindungen vorstellen, das Daten verarbeitet. Jedes Neuron ist wie ein kleiner Schalter, der entscheidet, welche Information weitergegeben wird.
Gewichte sind Zahlen, die bestimmen, wie stark eine Verbindung zwischen Neuronen ist. Wenn das Modell trainiert wird, passt es die Gewichte an, um bessere Entscheidungen zu treffen.
Am Anfang trifft das Modell viele Fehler, aber durch wiederholtes Training lernt es, welche Wörter typischerweise auf andere folgen.

### Wie genau sagt das Modell den nächsten Text vorher?

Das Modell betrachtet eine Sequenz von Wörtern und berechnet, welches Wort am wahrscheinlichsten als nächstes folgen sollte. Dies geschieht in mehreren Schritten:
Eingabetext in Tokens umwandeln – Der Text wird in numerische Werte umgewandelt.
Ein neuronales Netzwerk verarbeitet die Tokens – Die Token-Reihenfolge wird durch verschiedene Ebenen des Netzwerks verarbeitet.
Wahrscheinlichkeiten für das nächste Token berechnen – Das Modell gibt eine Liste möglicher nächster Tokens mit jeweiligen Wahrscheinlichkeiten aus.
Das wahrscheinlichste Token auswählen – Das Modell entscheidet sich für das wahrscheinlichste Wort und gibt es als nächstes Token aus.
Wiederholung bis zur maximalen Textlänge – Dieser Prozess wiederholt sich, bis die gewünschte Textlänge erreicht ist.

## 3. Was ist Hugging Face und warum nutzen wir es?

**Hugging Face** ist eine Plattform für Open-Source-KI-Modelle. Es stellt vortrainierte Modelle, Datensätze und Bibliotheken zur Verfügung, die die Arbeit mit LLMs erleichtern. In diesem Projekt nutzen wir:

- **Transformers**: Bibliothek für das Laden und Trainieren von vortrainierten Modellen.
- **Datasets**: Ermöglicht das Laden und Vorbereiten von Textdatensätzen.
- **Hugging Face Hub**: Eine Plattform, um Modelle hochzuladen und mit anderen zu teilen.

Das Modell wird mit GPT-2 von Hugging Face trainiert, aber es können auch andere Modelle genutzt werden. Alternativen sind:
- **GPT-2 Medium/Large/XL** – Größere Versionen von GPT-2 für bessere Ergebnisse.
- **GPT-Neo (EleutherAI)** – Ein Open-Source-Modell mit leistungsstärkeren Architekturen.
- **BLOOM** – Ein mehrsprachiges Open-Source-LLM.

## 4. Funktionen und Einschränkungen

### Funktionen

- **Texterzeugung**: Das Modell generiert Texte auf Basis von Benutzereingaben.
- **Anpassbarkeit**: Parameter wie Trainingsdauer, Token-Länge oder Batch-Größe sind individuell verstellbar.
- **Offline-Nutzung**: Kein API-Zugriff erforderlich, das Modell läuft lokal.
- **Eigene Trainingsdaten**: Beliebige Datensätze können für Feintuning genutzt werden.

### Einschränkungen

- **Kein langfristiger Kontext**: Begrenzte Fähigkeit, sich an vorherige Aussagen zu erinnern.
- **Beschränkte Trainingsdaten**: Das Modell lernt nur aus ca. 10.000 OpenWebText-Beispielen, der Parameter ist anpassbar.
- **Einfache Architektur**: Keine fortgeschrittenen Trainingsmethoden wie RLHF (Reinforcement Learning from Human Feedback).

## 5. Nutzung und Installation

### Projekt klonen und Umgebung vorbereiten

```bash
git clone https://github.com/dein-username/MINILLM.git
cd MINILLM
python3 -m venv venv
```
Je nach Betriebssystem muss die virtuelle Umgebung unterschiedlich aktiviert werden:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

### Abhängigkeiten installieren

Das Projekt benötigt einige Bibliotheken, die über `requirements.txt` installiert werden:
```bash
pip install -r requirements.txt
```
Diese beinhalten unter anderem:
- **transformers**: Bibliothek von Hugging Face zur Arbeit mit vortrainierten Sprachmodellen.
- **torch**: PyTorch für das Training von neuronalen Netzen.
- **datasets**: Hilfsbibliothek zum Laden und Verarbeiten von Datensätzen.
- **accelerate**: Optimiert das Training auf verschiedenen Hardware-Setups.
- **huggingface_hub**: Schnittstelle zur Interaktion mit der Hugging Face Plattform.

### Modell trainieren (Dauer je nach Hardware zwischen 1–6 Stunden)

```bash
python train_llm.py
```
In der Datei `train_llm.py` kann der Datensatz geändert werden, indem die Quelle in folgender Zeile angepasst wird:
```python
dataset = load_dataset("openwebtext", trust_remote_code=True)
```
Alternativ können hier eigene Trainingsdaten eingebunden werden, z. B.:
- **Wikipedia-Daten** (`wikipedia` über Hugging Face Datasets)
- **Common Crawl** (Web-Texte)
- **Persönliche Dokumente oder Spezialtexte**

## 6. Parameter zur Anpassung

| Parameter                     | Bedeutung                                         | Standardwert          | Alternative Werte |
| ----------------------------- | ------------------------------------------------- | --------------------- | ---------------- |
| `num_train_epochs`            | Anzahl der Trainingsdurchläufe                    | `2`                   | `1 - 10` (je nach Leistung) |
| `max_length`                  | Maximale Tokenlänge für Eingaben                  | `256`                 | `128, 512, 1024` |
| `per_device_train_batch_size` | Anzahl der Samples pro Trainingsschritt           | `2`                   | `1 - 16` (abhängig vom RAM) |
| `learning_rate`               | Lerngeschwindigkeit des Modells                   | `5e-5`                 | `1e-4, 3e-5` |
| `tokenizer.pad_token`         | Kein Standard-Pad-Token in GPT-2, daher EOS-Token | `tokenizer.eos_token` | `"[PAD]"` |

## 7. Fachbegriffe verständlich erklärt

- **Token**: Eine kleinste Einheit eines Textes (Wort oder Zeichen), die das Modell verarbeitet.
- **Tokenization**: Der Prozess, Text in einzelne Tokens umzuwandeln, bevor das Modell sie verarbeitet.
- **Batched Processing**: Training und Inferenz werden in Stapeln (`Batches`) verarbeitet, um Rechenleistung effizient zu nutzen.
- **Truncation**: Kürzung langer Eingaben, um die maximale Tokenanzahl einzuhalten.
- **Padding**: Auffüllen von zu kurzen Eingaben, um gleiche Längen zu gewährleisten.
- **Epochen**: Anzahl der Trainingsdurchläufe durch den gesamten Datensatz.
- **Pipeline**: Eine vereinfachte Schnittstelle von Hugging Face für verschiedene NLP-Aufgaben.
- **Dataset**: Eine Sammlung von Texten oder Daten, die für das Training genutzt werden.

## 8. Interessante Fakten über Künstliche Intelligenz (KI)

10-15 % der gesamten Rechenleistung in modernen LLMs wird verwendet, um Bias (Voreingenommenheit) in den generierten Texten zu minimieren.
Moderne KI-Modelle haben mehr Parameter als das menschliche Gehirn Synapsen besitzt – GPT-4 soll über eine Billion Parameter haben.
Datenqualität ist wichtiger als Datenmenge – Ein kleiner, sauberer Datensatz kann oft bessere Ergebnisse liefern als riesige Mengen unstrukturierter Daten.
KI benötigt ständiges Feintuning – Selbst fortgeschrittene Modelle müssen regelmäßig nachtrainiert werden, um mit neuen Informationen Schritt zu halten.

## 9. Weitere Entwicklungsmöglichkeiten

- Mehr Trainingsdaten für bessere Ergebnisse.
- Erstellung eines Web-Interfaces mit Gradio oder Streamlit.
- Hochladen des Modells auf Hugging Face.

Ich freue mich über Beiträge, Kritik und Kommentare.


