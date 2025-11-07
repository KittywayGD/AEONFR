# Guide d'Installation et Utilisation - Windows

Guide complet pour installer et utiliser le Recursive Code LLM sur Windows.

> ⚠️ **Installation échouée ?** Si `quick_start.bat` a rencontré des erreurs, lance simplement `fix_installation.bat` pour corriger l'installation automatiquement.

## 📋 Prérequis

### 1. Python 3.9+
Télécharge et installe Python depuis [python.org](https://www.python.org/downloads/)

**IMPORTANT lors de l'installation :**
- ✅ Coche "Add Python to PATH"
- ✅ Coche "Install pip"

Vérifie l'installation :
```cmd
python --version
pip --version
```

### 2. CUDA Toolkit (pour GPU NVIDIA)

Pour utiliser ta RTX 5060Ti, installe CUDA Toolkit :

1. Vérifie ta version de driver NVIDIA :
   ```cmd
   nvidia-smi
   ```

2. Télécharge CUDA Toolkit 11.8 ou 12.x depuis [nvidia.com/cuda](https://developer.nvidia.com/cuda-downloads)

3. Installe en suivant l'assistant

### 3. Visual Studio Build Tools (optionnel mais recommandé)

Certains packages Python nécessitent des outils de compilation :
- Télécharge [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
- Installe "Desktop development with C++"

## 🚀 Installation Rapide

### Option 1 : Script Automatique (Recommandé)

1. Ouvre un terminal (PowerShell ou CMD) dans le dossier du projet :
   ```cmd
   cd chemin\vers\AEONFR\coding-ai
   ```

2. Lance le script d'installation :
   ```cmd
   quick_start.bat
   ```

### Option 2 : Installation Manuelle

1. **Ouvre PowerShell ou CMD** dans le dossier `coding-ai`

2. **Crée un environnement virtuel** :
   ```cmd
   python -m venv venv
   ```

3. **Active l'environnement virtuel** :

   PowerShell :
   ```powershell
   venv\Scripts\Activate.ps1
   ```

   CMD :
   ```cmd
   venv\Scripts\activate.bat
   ```

   **Note** : Si PowerShell bloque l'exécution, tape :
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Mets à jour pip** :
   ```cmd
   python -m pip install --upgrade pip
   ```

5. **Installe les dépendances** :
   ```cmd
   pip install -r requirements.txt
   ```

   ⏱️ Cela peut prendre 5-10 minutes

6. **Vérifie l'installation de PyTorch avec CUDA** :
   ```cmd
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
   ```

   **Si CUDA n'est pas disponible**, réinstalle PyTorch avec CUDA :
   ```cmd
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

7. **Crée les dossiers nécessaires** :
   ```cmd
   mkdir checkpoints logs data
   ```

## 🎮 Utilisation

### Activer l'Environnement Virtuel

**À CHAQUE FOIS que tu ouvres un nouveau terminal**, active l'environnement :

PowerShell :
```powershell
venv\Scripts\Activate.ps1
```

CMD :
```cmd
venv\Scripts\activate.bat
```

Tu verras `(venv)` devant ton prompt quand c'est activé.

### Lancer l'Entraînement

```cmd
python train.py --config config\training_config.yaml
```

### Reprendre depuis un Checkpoint

```cmd
python train.py --config config\training_config.yaml --resume
```

### Arrêter l'Entraînement Proprement

Appuie sur `Ctrl+C` - le système sauvegarde automatiquement avant de s'arrêter.

### Générer du Code (Mode Interactif)

Après l'entraînement :
```cmd
python inference.py --model checkpoints\final_model\pytorch_model.bin --tokenizer checkpoints\tokenizer.json
```

### Lancer les Tests

```cmd
pytest tests\
```

## 📊 Monitoring

### Weights & Biases (W&B)

1. **Inscris-toi** sur [wandb.ai](https://wandb.ai)

2. **Connecte-toi** :
   ```cmd
   wandb login
   ```

3. **Active dans la config** (`config\training_config.yaml`) :
   ```yaml
   logging:
     use_wandb: true
     wandb_project: "recursive-code-llm"
     wandb_entity: "ton-username"
   ```

### TensorBoard

```cmd
tensorboard --logdir logs\tensorboard
```

Puis ouvre http://localhost:6006 dans ton navigateur.

## ⚙️ Configuration GPU

### Pour RTX 5060Ti (8GB VRAM)

Modifie `config\training_config.yaml` :

```yaml
model:
  hidden_size: 512          # Commence petit
  num_hidden_layers: 6
  num_attention_heads: 8

training:
  batch_size: 2
  gradient_accumulation_steps: 16
  mixed_precision: "fp16"   # Économise 50% de VRAM
  gradient_checkpointing: true
```

### Pour RTX 5060Ti (16GB VRAM)

```yaml
model:
  hidden_size: 768
  num_hidden_layers: 8
  num_attention_heads: 12

training:
  batch_size: 4
  gradient_accumulation_steps: 8
```

## 🐛 Résolution de Problèmes

### ⚡ Script d'Installation Rapide pour les Erreurs

Si `quick_start.bat` a échoué, lance simplement :
```cmd
fix_installation.bat
```

Ce script va :
1. Nettoyer les installations échouées
2. Installer PyTorch correctement (avec CUDA)
3. Installer le reste des dépendances dans le bon ordre

### Erreur : DeepSpeed installation failed

**C'est normal sur Windows !** DeepSpeed est difficile à installer sur Windows et n'est pas nécessaire pour démarrer.

**Solution** : Le fichier `requirements.txt` a été mis à jour pour rendre DeepSpeed optionnel. Relance simplement :
```cmd
fix_installation.bat
```

Ou manuellement :
```cmd
venv\Scripts\activate.bat
pip uninstall deepspeed
pip install -r requirements.txt
```

### Erreur : "CUDA out of memory"

**Solutions** :
1. Réduis `batch_size` à 1
2. Augmente `gradient_accumulation_steps` à 32
3. Réduis `hidden_size` (512 ou moins)
4. Active `gradient_checkpointing: true`

### Erreur : "torch not compiled with CUDA"

**Réinstalle PyTorch avec CUDA** :
```cmd
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Erreur : "ModuleNotFoundError"

**Active l'environnement virtuel** :
```cmd
venv\Scripts\activate.bat
```

### Scripts PowerShell bloqués

**Change la politique d'exécution** :
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Installation lente

**Utilise un miroir pip** :
```cmd
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Docker n'est pas disponible

**Désactive Docker dans la config** :
```yaml
recursive:
  evaluation:
    use_docker: false  # Utilise subprocess à la place
```

## 📁 Structure des Fichiers Windows

Les chemins sous Windows utilisent des backslashes (`\`) :

```
C:\Users\VotreNom\AEONFR\coding-ai\
├── config\
│   └── training_config.yaml
├── src\
│   ├── model\
│   ├── training\
│   ├── recursive\
│   └── data\
├── checkpoints\
├── logs\
├── venv\              # Environnement virtuel
├── train.py
└── quick_start.bat
```

## 💡 Astuces Windows

### 1. Utilise Windows Terminal

Plus moderne et pratique que CMD :
- Télécharge depuis le Microsoft Store
- Support des onglets
- Meilleur rendu des couleurs

### 2. Crée un Raccourci

Crée un fichier `start_training.bat` :
```batch
@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
python train.py --config config\training_config.yaml
pause
```

Double-clique dessus pour lancer l'entraînement !

### 3. Surveille ton GPU

Installe GPU-Z ou MSI Afterburner pour monitorer :
- Température
- Utilisation VRAM
- Clock speeds

### 4. Performance

**Désactive l'antivirus temporairement** pour les dossiers du projet (peut ralentir pip/training)

**Ferme les applications gourmandes** (jeux, Chrome avec 50 onglets, etc.)

## 🔥 Commandes Rapides

Copie-colle ces commandes utiles :

```cmd
REM Activer l'environnement
venv\Scripts\activate.bat

REM Entraîner
python train.py --config config\training_config.yaml

REM Reprendre
python train.py --resume

REM Générer du code
python inference.py --model checkpoints\final_model\pytorch_model.bin --tokenizer checkpoints\tokenizer.json

REM Voir les logs
type logs\training.log

REM Tester
pytest tests\ -v

REM Voir l'utilisation GPU
nvidia-smi
```

## 📞 Besoin d'Aide ?

1. **Erreurs Python** : Vérifie que l'environnement virtuel est activé
2. **Erreurs CUDA** : Vérifie `nvidia-smi` et réinstalle PyTorch
3. **Erreurs VRAM** : Réduis la taille du modèle dans la config
4. **Lenteur** : Vérifie que tu utilises bien le GPU avec CUDA

---

**Bon entraînement ! 🚀**
