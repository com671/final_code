# News Topic Classification with TF-IDF and CNN

An end-to-end text classification project using AG News. Given a news headline and description, the system predicts **World**, **Sports**, **Business**, or **Sci/Tech**.

The reported experiment compares TF-IDF with Logistic Regression against a text CNN and tests the effect of removing CNN dropout. All project code runs directly in a Jupyter or Google Colab notebook. No separate Python source files are required.

## Project structure

| File or directory | Purpose |
| --- | --- |
| `README.md` | Setup, execution instructions, and experiment overview. |
| `AG_News_Fast_Colab.ipynb` | Data preparation, training, validation, testing, inference, and result export. |
| `ag_news_fast_results/` | Generated configuration, metrics, predictions, plots, and audit files. |
| `ag_news_fast_results/raw/` | Downloaded dataset CSV files, cached for subsequent runs. |
| `AG_News_Fast_Results.zip` | Generated results archive, excluding the raw dataset. |

Upload the notebook alongside this README in the repository root. The results directory and ZIP are generated when the notebook runs.

## Data and experimental design

Inputs combine each article's headline and description. Labels are converted to integers from 0 to 3 in this order: World, Sports, Business, Sci/Tech.

| Split | Examples | Examples per class |
| --- | ---: | ---: |
| Training | 4,000 | 1,000 |
| Validation | 800 | 200 |
| Test | 1,000 | 250 |

Training and validation records come from the official training partition. Test records come from the official test partition. The notebook normalizes whitespace and HTML entities, removes conflicting or duplicate development records, checks development/test text overlap, and samples balanced splits.

Vocabularies and TF-IDF features are fitted on training data only. CNN checkpoints are selected using validation macro-F1. Test labels are reserved for final evaluation.

## Dependencies

Use Python 3.11 or 3.12. The notebook requires:

- PyTorch
- scikit-learn
- NumPy and pandas
- Matplotlib
- IPython and threadpoolctl
- Transformers 4.57.1 for the notebook's optional Transformer branch
- Jupyter Notebook and nbconvert for local execution

Google Colab supplies most dependencies. The first notebook cell installs PyTorch or Transformers if missing. Internet access is required to install packages and download the dataset. The optional Transformer branch also downloads pretrained model files.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Upload notebook** and upload `AG_News_Fast_Colab.ipynb`.
3. A CPU runtime is sufficient. A GPU is optional.
4. For the CNN comparison reported below, configure Section 5 as described under **Select the reported experiment** before running all cells.
5. Select **Runtime > Run all**.
6. Review the displayed metrics, confusion matrices, error examples, and prediction demonstration.
7. Download `AG_News_Fast_Results.zip` when prompted, or use Colab's Files panel.

In Colab, generated results are written to `/content/ag_news_fast_results/`.

## Select the reported experiment

The supplied notebook contains an optional BERT-Tiny feature-extraction stage in Section 5. Its original behavior runs the CNN dropout comparison if that stage fails or exceeds its configured budget. A successful Transformer run therefore produces a different model comparison.

To explicitly run the reported **TF-IDF / CNN / CNN without dropout** experiment, replace the Section 5 code cell with the following before selecting **Run all**. Keep the preceding data, baseline, and CNN cells and all subsequent evaluation cells.

```python
# Explicitly select the CNN dropout comparison.
BERT_INFO = {
    "status": "not_run",
    "reason": "CNN dropout ablation selected for this experiment"
}
FEATURES = {}

fit_cnn("CNN_no_dropout", dropout=0.0)

ABLATION = {
    "a": "CNN",
    "b": "CNN_no_dropout",
    "changed": "dropout 0.5 versus 0.0",
    "controlled": (
        "same initialization seed, training order, architecture, "
        "data, optimizer and epoch budget"
    )
}

save_json("transformer_details.json", BERT_INFO)
save_json("ablation_design.json", ABLATION)
save_json("validation_metrics.json", VALIDATION)
save_json("failures.json", FAILURES)
pd.DataFrame(HISTORY).to_csv(OUT / "training_history.csv", index=False)
```

This selection prevents the reported model set from depending on whether a pretrained model download succeeds. Leave the optional Transformer cell unchanged only if you intend to run that separate comparison.

## Run locally

From the repository root, create an environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install "torch>=2.6" "transformers==4.57.1" scikit-learn numpy pandas matplotlib ipython threadpoolctl notebook nbconvert
```

Open the notebook:

```bash
jupyter notebook AG_News_Fast_Colab.ipynb
```

After selecting the reported experiment as described above, run all cells in order. Alternatively, execute the configured notebook from the command line:

```bash
jupyter nbconvert --to notebook --execute AG_News_Fast_Colab.ipynb --ExecutePreprocessor.timeout=-1 --output AG_News_Executed.ipynb
```

Locally, outputs are saved under `ag_news_fast_results/` in the working directory. Runtime depends on hardware and package versions.

## Models and training

**TF-IDF + Logistic Regression:** Word unigrams and bigrams, up to 20,000 features, minimum document frequency 2, sublinear term frequency, and Logistic Regression with `C=1`.

**Text CNN:** A training-only vocabulary of up to 12,000 tokens, 64-dimensional trainable embeddings, convolution widths 3/4/5 with 32 filters each, ReLU, masked max pooling, and a four-class output layer. Inputs are capped at 64 tokens.

**Dropout ablation:** Compare dropout 0.5 against dropout 0.0 while holding other settings consistent.

CNN training uses Adam with learning rate 0.002, batch size 64, cross-entropy loss, gradient clipping, and three epochs. The best validation macro-F1 checkpoint is restored. The training seed is 42. Configuration values are defined in `CFG` near the start of the notebook.

## Evaluation and reported results

The primary metric is **macro-F1**, supported by accuracy, class-level precision/recall/F1, confusion matrices, and prediction timing.

The submitted run produced the following results:

| Model | Validation macro-F1 | Test macro-F1 | Test accuracy |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.8781 | 0.8765 | 87.7% |
| CNN | 0.6699 | 0.6619 | 65.9% |
| CNN without dropout | 0.6977 | 0.7117 | 71.6% |

These values describe the submitted run, rather than guaranteed outputs on every environment. The TF-IDF baseline performed best. Removing dropout improved overall CNN performance, but changes differed by category, especially between Business and Sci/Tech.

## Generated outputs

Important files inside `ag_news_fast_results/` include:

| Output | Contents |
| --- | --- |
| `results.csv` | Validation/test scores and measured prediction times. |
| `test_metrics.json` | Class-level metrics and confusion matrices. |
| `predictions_*.csv` | Record-level true labels, predictions, and probabilities. |
| `training_history.csv` | CNN epoch losses and validation metrics. |
| `confusion_matrices.png` | Category-level errors for the completed models. |
| `learning_curves.png` | Training losses and validation macro-F1. |
| `model_comparison.png` | Test macro-F1 comparison. |
| `error_analysis.csv` | Incorrect predictions sampled for manual analysis. |
| `config.json`, `environment.json` | Settings and runtime package versions. |
| `data_audit.json`, `selected_data.csv` | Dataset provenance and selected records. |
| `report_data.json` | Consolidated evidence for interpreting the run. |

The export cell packages the generated evidence into `AG_News_Fast_Results.zip`.

## Predict a new article

After running the training cells, edit the example in Section 8 or run:

```python
predict_article(
    "National team wins football championship",
    "The players celebrated their victory in the tournament final."
)
```

The function returns the selected model, predicted topic, and uncalibrated class probabilities. In the CNN experiment, this demonstration uses the standard CNN. Keep the notebook session active: trained models are held in memory, and the results ZIP is not a saved-model deployment package.

## Limitations and responsible use

Results use sampled data and one training seed, with no estimate of variability across training seeds. AG News contains historical English news and four broad categories, so results do not establish performance on current news, other languages, or additional topics. Exact duplicate checks do not remove every related story.

Topic predictions do not verify factual accuracy. Probabilities are uncalibrated, and ambiguous articles can receive incorrect labels. Practical use should support human correction and evaluate performance on the intended news source.

## References

- [AG News dataset description](https://www.tensorflow.org/datasets/catalog/ag_news_subset)
- [Dataset CSV source used by the notebook](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)
- [Understanding Deep Learning, Simon J. D. Prince](https://udlbook.github.io/udlbook/)

