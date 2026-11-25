# Machine learning

| Component | Purpose | Status in this snapshot |
|---|---|---|
| [Evidence classifier](evidence_classifier/README.md) | Classifies the relationship between a claim and supplied evidence. The installation uses `p_supports` as the machine's agreement score. | Training code and FastAPI endpoint are present. Restore or train the local E06b artifact to start the API. |
| [Intent classifier](intent_classifier/README.md) | Classifies short visitor utterances so the machine mask can select a spoken reaction. | Training plan. Labeled data, training code, model artifact and integration are pending. |

Python dependencies are listed in [requirements.txt](requirements.txt). Raw datasets, processed datasets, generated experiments and local model bundles are excluded by the root `.gitignore`.
