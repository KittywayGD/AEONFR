# Recursive Code LLM

Un modèle de langage orienté programmation qui s'entraîne de manière récursive et autonome sur GPU RTX 5060Ti.

## 🎯 Objectif

Ce projet implémente un système d'apprentissage autonome où un modèle de langage :
1. Génère du code Python
2. Exécute et évalue le code généré
3. Ajoute les solutions valides au dataset d'entraînement
4. S'améliore continuellement en apprenant de ses propres générations réussies

## 🏗️ Architecture

### Modèle
- **Architecture** : Transformer decoder-only (style GPT)
- **Taille** : ~100M-500M paramètres (optimisé pour 8-16GB VRAM)
- **Couches** : 6-12 transformer blocks
- **Hidden size** : 512-1024
- **Attention heads** : 8-16
- **Optimisations** : Mixed precision (FP16/BF16), gradient checkpointing, gradient accumulation

### Composants Clés

1. **Tokenizer Custom BPE** : Optimisé pour le code Python
2. **Système de Checkpointing Robuste** : Sauvegarde automatique avec pause/reprise
3. **Boucle d'Auto-amélioration** : Génération → Évaluation → Validation → Ajout au dataset
4. **Sandbox d'Exécution** : Exécution sécurisée du code généré (subprocess/Docker)
5. **Gestion de Dataset Dynamique** : Versioning et stockage des exemples validés

## 📁 Structure du Projet

```
coding-ai/
├── config/
│   └── training_config.yaml       # Configuration complète
├── src/
│   ├── model/
│   │   ├── architecture.py        # Modèle Transformer
│   │   └── tokenizer.py           # Tokenizer BPE custom
│   ├── training/
│   │   ├── trainer.py             # Boucle d'entraînement
│   │   └── checkpoint.py          # Gestion des checkpoints
│   ├── recursive/
│   │   ├── generator.py           # Génération de code
│   │   ├── evaluator.py           # Évaluation et exécution
│   │   └── feedback_loop.py       # Boucle récursive
│   └── data/
│       └── dataset.py             # Gestion des datasets
├── checkpoints/                   # Sauvegardes du modèle
├── logs/                          # Logs d'entraînement
├── tests/                         # Tests unitaires
├── train.py                       # Point d'entrée principal
├── requirements.txt               # Dépendances
└── README.md                      # Ce fichier
```

## 🚀 Installation

### Prérequis
- Python 3.9+
- CUDA 11.8+ (pour GPU)
- 8-16GB VRAM GPU (RTX 5060Ti)
- ~50GB d'espace disque

### Installation des dépendances

```bash
cd coding-ai

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) Flash Attention pour meilleures performances
# pip install flash-attn --no-build-isolation
```

## ⚙️ Configuration

Le fichier `config/training_config.yaml` contient tous les paramètres configurables :

### Paramètres Clés à Ajuster

```yaml
model:
  hidden_size: 768          # 512, 768, ou 1024 selon VRAM
  num_hidden_layers: 8      # 6-12 layers
  num_attention_heads: 12   # 8-16 heads

training:
  batch_size: 2             # Augmenter si VRAM > 8GB
  gradient_accumulation_steps: 16  # Effective batch = 32
  learning_rate: 3.0e-4
  num_epochs: 10

checkpoint:
  save_steps: 500           # Checkpoint tous les N steps
  save_time_interval: 1800  # Checkpoint tous les 30 min
  keep_last_n: 3            # Garder 3 derniers checkpoints

recursive:
  enabled: true             # Activer l'apprentissage récursif
  start_after_steps: 5000   # Commencer après 5000 steps
  generation_interval: 1000 # Générer tous les 1000 steps
```

## 🏃 Utilisation

### Entraînement de Base

```bash
# Démarrer l'entraînement
python train.py --config config/training_config.yaml

# Reprendre depuis le dernier checkpoint
python train.py --config config/training_config.yaml --resume

# Utiliser un tokenizer pré-entraîné
python train.py --config config/training_config.yaml --tokenizer checkpoints/tokenizer.json
```

### Pause et Reprise

Le système gère automatiquement les interruptions :

```bash
# Pendant l'entraînement, presser Ctrl+C pour une interruption propre
# Le système sauvegarde automatiquement l'état complet

# Reprendre plus tard
python train.py --resume
```

### Monitoring

#### W&B (Weights & Biases)
```bash
# Activer dans config/training_config.yaml
logging:
  use_wandb: true
  wandb_project: "recursive-code-llm"
  wandb_entity: "votre-username"

# Se connecter à W&B
wandb login
```

#### TensorBoard
```bash
# Lancer TensorBoard
tensorboard --logdir logs/tensorboard
```

## 📊 Métriques Suivies

### Entraînement
- **Loss** : Cross-entropy loss
- **Perplexity** : Exp(loss)
- **Learning Rate** : Taux d'apprentissage courant
- **Epoch & Global Step** : Progression

### Apprentissage Récursif
- **Generation Rate** : Nombre de samples générés
- **Success Rate** : % de samples valides
- **Quality Score** : Score de qualité moyen
- **Dataset Size** : Taille du dataset dynamique
- **Execution Success** : % d'exécutions réussies

## 🔧 Optimisations pour RTX 5060Ti

Le projet est optimisé pour fonctionner efficacement sur des GPU avec VRAM limitée :

1. **Mixed Precision (FP16)** : Réduit l'utilisation mémoire de ~50%
2. **Gradient Accumulation** : Simule des batch sizes plus grands
3. **Gradient Checkpointing** : Économise la mémoire au coût de ~20% de temps
4. **Small Batch Sizes** : batch_size=2 avec accumulation
5. **Efficient Attention** : Support optionnel de Flash Attention

### Estimation VRAM

| Config | Hidden Size | Layers | Parameters | VRAM (Train) | VRAM (Infer) |
|--------|-------------|--------|------------|--------------|--------------|
| Tiny   | 512         | 6      | ~100M      | ~6 GB        | ~2 GB        |
| Small  | 768         | 8      | ~200M      | ~8 GB        | ~3 GB        |
| Medium | 1024        | 12     | ~500M      | ~14 GB       | ~4 GB        |

## 🔐 Sécurité du Sandbox

Le code généré est exécuté dans un environnement isolé :

### Mode Subprocess (Par défaut)
- Exécution dans un processus séparé
- Timeout configurable (10s par défaut)
- Pas d'accès réseau
- Ressources limitées

### Mode Docker (Recommandé pour production)
```yaml
recursive:
  evaluation:
    use_docker: true
    docker_image: "python:3.10-slim"
```

Avantages :
- Isolation complète
- Limites mémoire/CPU strictes
- Pas d'accès filesystem

## 📈 Boucle d'Auto-amélioration

### Fonctionnement

1. **Génération** : Le modèle génère du code à partir de prompts
2. **Évaluation** :
   - Vérification de syntaxe (AST parsing)
   - Exécution dans sandbox
   - Calcul de métriques qualité
3. **Filtrage** : Seuls les samples avec score > seuil sont gardés
4. **Ajout au Dataset** : Les bons samples enrichissent le dataset
5. **Fine-tuning** : Entraînement périodique sur nouvelles données

### Prompts Générés

Le système génère automatiquement des prompts variés :
- Définitions de fonctions
- Implémentations de classes
- Algorithmes classiques
- Code avec docstrings
- Résolution de problèmes

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_model.py
pytest tests/test_tokenizer.py
pytest tests/test_evaluator.py

# Avec couverture
pytest --cov=src tests/
```

## 🐛 Debugging

### Logs

Les logs sont sauvegardés dans `logs/training.log` :

```bash
# Suivre les logs en temps réel
tail -f logs/training.log

# Rechercher des erreurs
grep ERROR logs/training.log
```

### Mode Debug

```yaml
logging:
  log_level: "DEBUG"  # Plus de détails dans les logs
```

### Checkpoints Corrompus

```bash
# Lister les checkpoints disponibles
ls -lh checkpoints/

# Charger un checkpoint spécifique
# Modifier checkpoint_manager.load_checkpoint() avec le chemin
```

## 🚧 Limitations Connues

1. **Python uniquement** : Pour l'instant, seul Python est supporté
2. **Tests limités** : Pas de génération automatique de tests unitaires
3. **Qualité variable** : Les premières itérations génèrent du code simple
4. **Compute intensif** : L'entraînement complet peut prendre plusieurs jours

## 🗺️ Roadmap

- [ ] Support multi-langages (JavaScript, Go, Rust)
- [ ] Génération automatique de tests
- [ ] Évaluation basée sur des benchmarks (HumanEval, MBPP)
- [ ] Fine-tuning avec RLHF
- [ ] Interface web pour monitoring
- [ ] Quantization (INT8/INT4) pour inference
- [ ] Support multi-GPU

## 📚 Références

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Decoder-only LM
- [CodeGen](https://arxiv.org/abs/2203.13474) - Code generation models
- [Self-Instruct](https://arxiv.org/abs/2212.10560) - Self-improvement approach

## 🤝 Contribution

Les contributions sont bienvenues ! N'hésitez pas à :
- Ouvrir des issues pour bugs ou suggestions
- Proposer des PRs pour nouvelles fonctionnalités
- Améliorer la documentation

## 📄 Licence

MIT License - Voir LICENSE pour détails

## 🙏 Remerciements

- HuggingFace pour les outils de tokenization et datasets
- PyTorch team pour le framework
- Accelerate pour l'entraînement distribué optimisé

---

**Note** : Ce projet est expérimental et destiné à la recherche. N'utilisez pas le code généré en production sans validation humaine approfondie.
