import os


def main():
    """Run all the features/targets combinations for RECOLA datasets.
    Features are either Mel Filter-Bank (MFB) or wav2vec2 representations trained for French data (LeBenchmark)
    Targets are arousal and valence dimensions of emotion.
    """
    data_path = "/data/datasets/original/RECOLA_2016"
    experiment_folder = "./Results_test"
    targets = ["arousal", "valence"]
    features = [
        "MFB",
        "LeBenchmark/wav2vec2-FR-1K-base",
        "LeBenchmark/wav2vec2-FR-2.6K-base",
        "LeBenchmark/wav2vec2-FR-3K-base",
        "LeBenchmark/wav2vec2-FR-7K-base",
        "LeBenchmark/wav2vec2-FR-1K-large",
        "LeBenchmark/wav2vec2-FR-3K-large",
        "LeBenchmark/wav2vec2-FR-7K-large",
    ]

    for target in targets:
        for feature in features:
            if "MFB" in feature:
                settings_file = "settings_MFB.yaml"
                feat_size = 40
                os.system(
                    f"python train.py {settings_file} --emotion_dimension={target} --feat_size={feat_size} --experiment_folder={experiment_folder} --data_path={data_path}"
                )
            elif "wav2vec2" in feature:
                settings_file = "settings.yaml"
                if "base" in feature:
                    feat_size = 768
                elif "large" in feature:
                    feat_size = 1024
                os.system(
                    f"python train.py {settings_file} --emotion_dimension={target} --feat_size={feat_size} --experiment_folder={experiment_folder} --data_path={data_path} --w2v2_hub={feature}"
                )


if __name__ == "__main__":
    main()
