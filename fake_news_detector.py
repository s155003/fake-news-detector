import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              roc_curve, precision_recall_curve, average_precision_score,
                              f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import re
import string
import warnings
warnings.filterwarnings("ignore")


RANDOM_SEED  = 42
TEST_SIZE    = 0.2
MAX_FEATURES = 10000
NGRAM_RANGE  = (1, 2)
THRESHOLD    = 0.5


REAL_HEADLINES = [
    "Federal Reserve raises interest rates by 0.25 percent",
    "NASA confirms water ice found on the Moon's surface",
    "Scientists develop new cancer treatment showing promising results in trials",
    "Stock markets fall amid concerns over global trade tensions",
    "United Nations calls for ceasefire in ongoing conflict",
    "Apple announces new iPhone with improved battery life",
    "Hurricane makes landfall causing widespread flooding",
    "Congress passes bipartisan infrastructure spending bill",
    "Unemployment rate falls to lowest level in decade",
    "WHO reports declining COVID-19 cases globally",
    "Supreme Court rules on landmark privacy case",
    "New study links air pollution to increased dementia risk",
    "Electric vehicle sales surpass one million units this year",
    "Police arrest suspect in connection with bank robbery",
    "Local school district receives federal funding for new programs",
    "Scientists discover new species of deep sea creature",
    "City council votes to increase minimum wage",
    "Hospital reports breakthrough in organ transplant procedure",
    "Tech company lays off thousands of workers amid restructuring",
    "Government releases annual budget with focus on healthcare",
    "Olympic committee announces host city for 2032 games",
    "New climate report warns of accelerating sea level rise",
    "Pharmaceutical company recalls medication over contamination concerns",
    "International trade deal signed between three nations",
    "Research shows meditation reduces stress and anxiety",
    "Major airline cancels flights due to severe weather",
    "Scientists achieve record efficiency in solar panel technology",
    "Housing prices rise for fifth consecutive quarter",
    "Senate confirms new ambassador to key allied nation",
    "Study finds regular exercise improves mental health outcomes",
    "New legislation aims to protect children online",
    "Central bank holds interest rates steady amid economic uncertainty",
    "Researchers identify gene linked to rare inherited disease",
    "Fire damages historic building in downtown area",
    "Government announces new renewable energy targets for 2030",
    "Tech giant faces antitrust investigation by regulators",
    "Scientists confirm hottest year on record globally",
    "Voters approve ballot measure expanding public transit",
    "International space station receives new crew members",
    "Report finds plastic pollution increasing in ocean ecosystems",
]

FAKE_HEADLINES = [
    "BREAKING: Government admits to hiding alien contact for decades",
    "Secret cure for cancer suppressed by pharmaceutical companies",
    "Scientists PROVE vaccines cause autism in new bombshell study",
    "EXPOSED: Elite global cabal controls world food supply",
    "Obama secretly built tunnels under the White House",
    "5G towers confirmed to spread viruses by independent researchers",
    "Deep state operatives caught rigging election results",
    "Miracle pill dissolves belly fat overnight without diet or exercise",
    "SHOCKING: Moon landing was filmed in Hollywood studio",
    "Chemtrails confirmed as government mind control program",
    "Billionaire globalists planning massive population reduction scheme",
    "CIA document reveals they created AIDS to target minority communities",
    "URGENT: New world order planning to microchip all citizens by 2025",
    "Hollywood elites running child trafficking ring from pizza restaurant",
    "Natural remedy cures all cancers doctors do not want you to know",
    "BOMBSHELL: Climate change data completely fabricated by scientists",
    "Fluoride in water supply used to lower IQ of population",
    "Secret society controls every major government on the planet",
    "PROOF: Election machines connected to foreign servers on election night",
    "Mainstream media ordered to hide truth about alien invasion",
    "George Soros funding riots in every major American city",
    "REVEALED: Coronavirus created in lab and released on purpose",
    "President signs secret executive order banning all firearms next month",
    "Banks planning to eliminate cash and control all spending by 2024",
    "Harvard study finds marijuana cures stage four cancer instantly",
    "LEAKED: Government plan to confiscate gold from private citizens",
    "Microchips found inside COVID vaccines by independent lab",
    "Shadow government controlling puppet presidents for fifty years",
    "EXCLUSIVE: FBI covered up evidence of massive voter fraud",
    "New research proves eating chocolate causes weight loss not gain",
    "Reptilian humanoids confirmed to hold key government positions",
    "BREAKING: United Nations planning military takeover of United States",
    "Suppressed Tesla technology would give everyone free energy",
    "Hollywood using movies to brainwash public into accepting socialism",
    "SHOCK STUDY: Cell phones emit radiation that sterilizes men",
    "Whistleblower exposes secret chemotherapy alternative hidden by doctors",
    "Deep state planning false flag attack to justify martial law",
    "REVEALED: Soros funded thousands of migrant caravans to invade country",
    "Ancient pyramids prove advanced alien civilization lived on Earth",
    "URGENT WARNING: New law will allow government to seize private property",
]

REAL_BODIES = [
    "Officials confirmed the data through multiple independent verification processes before releasing the findings to the public.",
    "The peer-reviewed study was published in a leading scientific journal and has since been replicated by researchers at three independent institutions.",
    "Authorities responded quickly to the situation, deploying emergency personnel and issuing guidance to affected residents.",
    "The legislation passed with bipartisan support following months of negotiation and compromise between both parties.",
    "Economists caution that while the figures are encouraging, longer-term structural challenges remain unresolved.",
    "The announcement follows years of research and development, with clinical trials showing statistically significant improvements.",
    "Market analysts noted the figures represent the strongest performance in the sector since the previous economic cycle.",
    "The decision was made after extensive consultation with stakeholders, experts, and affected communities.",
    "Environmental groups praised the move while industry representatives said they needed more time to assess the impact.",
    "The report draws on data collected from over forty countries spanning a fifteen-year period.",
]

FAKE_BODIES = [
    "This information is being suppressed by mainstream media. Share before it gets deleted. Wake up people the truth is out there.",
    "A source who asked to remain anonymous for fear of their life revealed the full extent of the conspiracy to this reporter.",
    "Doctors and scientists who dare speak the truth are being silenced, fired, and in some cases disappearing entirely.",
    "The globalist elite do not want you to know this. They have been hiding this information from the public for decades.",
    "Independent researchers who have studied this extensively confirm what the mainstream refuses to acknowledge.",
    "You will not see this on CNN or Fox News. Share this everywhere before the government shuts it down permanently.",
    "Thousands of whistleblowers have come forward but the deep state continues to cover up the undeniable evidence.",
    "This has been confirmed by multiple unnamed insiders who risk everything to bring you the real story.",
    "The pharmaceutical companies have known about this for years and have actively worked to hide the truth from patients.",
    "Patriots and truth-seekers everywhere are urged to spread this information before the internet censors remove it.",
]


def generate_dataset(n_real=500, n_fake=500):
    np.random.seed(RANDOM_SEED)

    def build_articles(headlines, bodies, label, n):
        articles = []
        for i in range(n):
            h = headlines[i % len(headlines)]
            b = bodies[i % len(bodies)]
            variation = f" {np.random.choice(['Report', 'Officials say', 'Sources confirm', 'Experts warn', 'Study finds', ''])} "
            text = f"{h}{variation}{b} " * np.random.randint(1, 4)
            articles.append({"text": text.strip(), "label": label, "headline": h})
        return articles

    real_articles = build_articles(REAL_HEADLINES, REAL_BODIES, 0, n_real)
    fake_articles = build_articles(FAKE_HEADLINES, FAKE_BODIES, 1, n_fake)

    df = pd.DataFrame(real_articles + fake_articles)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_text_features(df):
    df = df.copy()
    df["text_clean"]      = df["text"].apply(clean_text)
    df["char_count"]      = df["text"].apply(len)
    df["word_count"]      = df["text"].apply(lambda x: len(x.split()))
    df["exclamation"]     = df["text"].apply(lambda x: x.count("!"))
    df["question_marks"]  = df["text"].apply(lambda x: x.count("?"))
    df["caps_ratio"]      = df["text"].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    df["unique_words"]    = df["text_clean"].apply(lambda x: len(set(x.split())))
    df["lexical_density"] = df["unique_words"] / (df["word_count"] + 1)

    SENSATIONAL = ["bombshell", "shocking", "exposed", "urgent", "breaking",
                   "secret", "suppressed", "revealed", "proof", "confirmed",
                   "wake up", "deep state", "globalist", "elite", "cabal",
                   "cover up", "mainstream media", "they don't want you",
                   "share before", "truth", "whistleblower"]
    df["sensational_count"] = df["text"].apply(
        lambda x: sum(1 for w in SENSATIONAL if w in x.lower())
    )
    return df


def evaluate(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred)
    print(f"\n  {name}")
    print(f"  {'─'*35}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    return {"accuracy": acc, "auc": auc, "ap": ap, "f1": f1}


def plot_results(y_test, results, df, best_name):
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Fake News Detector — ML Analysis", fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(3, 3, 1)
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_test, data["probs"])
        ax1.plot(fpr, tpr, label=f"{name} (AUC={data['scores']['auc']:.3f})", linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(3, 3, 2)
    for name, data in results.items():
        prec, rec, _ = precision_recall_curve(y_test, data["probs"])
        ax2.plot(rec, prec, label=f"{name} (AP={data['scores']['ap']:.3f})", linewidth=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(3, 3, 3)
    cm = confusion_matrix(y_test, results[best_name]["preds"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax3,
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax3.set_title(f"Confusion Matrix ({best_name})")
    ax3.set_ylabel("Actual")
    ax3.set_xlabel("Predicted")

    ax4 = fig.add_subplot(3, 3, 4)
    best_probs  = results[best_name]["probs"]
    real_probs  = best_probs[y_test == 0]
    fake_probs  = best_probs[y_test == 1]
    ax4.hist(real_probs, bins=30, alpha=0.6, color="#2ca02c", label="Real News", density=True)
    ax4.hist(fake_probs, bins=30, alpha=0.6, color="#d62728", label="Fake News", density=True)
    ax4.axvline(THRESHOLD, color="black", linestyle="--", linewidth=2, label=f"Threshold ({THRESHOLD})")
    ax4.set_xlabel("Fake Probability Score")
    ax4.set_ylabel("Density")
    ax4.set_title("Score Distribution by Class")
    ax4.legend()

    ax5 = fig.add_subplot(3, 3, 5)
    model_names = list(results.keys())
    accs  = [results[n]["scores"]["accuracy"] for n in model_names]
    colors = ["#d62728" if n == best_name else "#1f77b4" for n in model_names]
    bars = ax5.bar(model_names, accs, color=colors)
    ax5.set_ylim(min(accs) - 0.05, 1.0)
    ax5.set_ylabel("Accuracy")
    ax5.set_title("Model Comparison — Accuracy")
    for bar, v in zip(bars, accs):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{v:.3f}", ha="center", fontsize=9)

    ax6 = fig.add_subplot(3, 3, 6)
    features = ["exclamation", "caps_ratio", "sensational_count", "lexical_density", "word_count"]
    real_df  = df[df["label"] == 0]
    fake_df  = df[df["label"] == 1]
    x = np.arange(len(features))
    w = 0.35
    real_means = [real_df[f].mean() for f in features]
    fake_means = [fake_df[f].mean() for f in features]
    norm_real  = [v / (max(real_means[i], fake_means[i]) + 1e-6) for i, v in enumerate(real_means)]
    norm_fake  = [v / (max(real_means[i], fake_means[i]) + 1e-6) for i, v in enumerate(fake_means)]
    ax6.bar(x - w/2, norm_real, w, label="Real News", color="#2ca02c", alpha=0.8)
    ax6.bar(x + w/2, norm_fake, w, label="Fake News", color="#d62728", alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(["Exclamations", "Caps Ratio", "Sensational\nWords", "Lexical\nDensity", "Word Count"], fontsize=8)
    ax6.set_title("Feature Comparison: Real vs Fake")
    ax6.set_ylabel("Normalized Mean Value")
    ax6.legend()

    ax7 = fig.add_subplot(3, 3, 7)
    ax7.boxplot([real_probs, fake_probs], labels=["Real News", "Fake News"],
                patch_artist=True,
                boxprops=dict(facecolor="#2ca02c", alpha=0.5),
                medianprops=dict(color="black", linewidth=2))
    ax7.set_ylabel("Predicted Fake Probability")
    ax7.set_title("Score Distribution Boxplot")

    ax8 = fig.add_subplot(3, 3, 8)
    thresholds = np.linspace(0.01, 0.99, 100)
    f1s  = [f1_score(y_test, (best_probs >= t).astype(int), zero_division=0) for t in thresholds]
    accs = [accuracy_score(y_test, (best_probs >= t).astype(int)) for t in thresholds]
    ax8.plot(thresholds, f1s,  label="F1 Score",  linewidth=2)
    ax8.plot(thresholds, accs, label="Accuracy",  linewidth=2)
    ax8.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={THRESHOLD}")
    ax8.set_xlabel("Decision Threshold")
    ax8.set_ylabel("Score")
    ax8.set_title("F1 and Accuracy vs Threshold")
    ax8.legend(fontsize=8)

    ax9 = fig.add_subplot(3, 3, 9)
    label_counts = df["label"].value_counts()
    ax9.pie([label_counts[0], label_counts[1]],
            labels=["Real News", "Fake News"],
            autopct="%1.1f%%",
            colors=["#2ca02c", "#d62728"],
            startangle=90)
    ax9.set_title("Dataset Class Distribution")

    plt.tight_layout()
    plt.savefig("fake_news_detector.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved as fake_news_detector.png")


def predict_article(pipeline, text, threshold=THRESHOLD):
    cleaned  = clean_text(text)
    prob     = pipeline.predict_proba([cleaned])[0][1]
    flagged  = prob >= threshold
    verdict  = "FAKE NEWS" if flagged else "REAL NEWS"
    conf     = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low"

    print(f"\n{'='*50}")
    print(f"  ARTICLE ANALYSIS")
    print(f"{'='*50}")
    print(f"  Text:       {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Fake Score: {prob:.4f} ({prob*100:.1f}%)")
    print(f"  Verdict:    {verdict}")
    print(f"  Confidence: {conf}")
    print(f"{'='*50}\n")
    return prob, flagged


print("Generating dataset...")
df = generate_dataset(n_real=500, n_fake=500)
print(f"Dataset: {len(df):,} articles | Real: {(df['label']==0).sum()} | Fake: {(df['label']==1).sum()}")

df = add_text_features(df)

X = df["text_clean"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

pipelines = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, sublinear_tf=True)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, sublinear_tf=True)),
        ("clf",   RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED))
    ]),
    "Gradient Boosting": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, sublinear_tf=True)),
        ("clf",   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                              max_depth=4, random_state=RANDOM_SEED))
    ]),
}

print(f"\n{'='*50}")
print(f"  MODEL EVALUATION")
print(f"{'='*50}")

results = {}
for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    probs  = pipeline.predict_proba(X_test)[:, 1]
    preds  = (probs >= THRESHOLD).astype(int)
    scores = evaluate(name, y_test, preds, probs)
    results[name] = {"pipeline": pipeline, "probs": probs, "preds": preds, "scores": scores}

best_name     = max(results, key=lambda k: results[k]["scores"]["auc"])
best_pipeline = results[best_name]["pipeline"]
print(f"\n  Best model: {best_name} (AUC: {results[best_name]['scores']['auc']:.4f})")

print("\nFull classification report (best model):")
print(classification_report(y_test, results[best_name]["preds"], target_names=["Real", "Fake"]))

plot_results(y_test, results, df, best_name)

real_sample = "Scientists publish peer-reviewed study showing new drug reduces tumor size in clinical trials"
fake_sample = "BOMBSHELL: Government hiding cure for all cancers to protect pharmaceutical profits SHARE NOW"

print("--- Real News Sample ---")
predict_article(best_pipeline, real_sample)

print("--- Fake News Sample ---")
predict_article(best_pipeline, fake_sample)

print("To classify your own article, call:")
print("  predict_article(best_pipeline, 'your article text here')")
print("\nTraining complete. Output saved to fake_news_detector.png")
