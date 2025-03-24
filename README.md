# Snake-Net

Un projet de réseau neuronal qui apprend à générer un jeu de Snake. Le modèle est entraîné à produire la matrice du jeu suivante en fonction de l'état actuel et de la direction choisie.

## Architecture

L'architecture du projet est la suivante :

- **CNN Encodeur** : Transforme la matrice 32×32 du jeu en une représentation latente de dimension 1024
- **Module d'Attention** : Intègre la représentation latente avec les 4 valeurs directionnelles (haut, droite, bas, gauche)
- **MLP** : Traite la sortie de l'attention et transforme la représentation
- **CNN Décodeur** : Retransforme le vecteur latent en une matrice 32×32 représentant le nouvel état du jeu

## Prérequis

- Python 3.8+
- PyTorch
- Pygame
- Numpy
- Matplotlib

Vous pouvez installer les dépendances avec :

```bash
pip install torch numpy pygame matplotlib
```

## Structure du projet

- `snake_game.py` : Implémentation du jeu Snake et génération de datasets
- `snake_model.py` : Définition du modèle SnakeNet avec CNN, attention et MLP
- `train.py` : Fonctions pour entraîner et évaluer le modèle
- `main.py` : Point d'entrée principal avec interface en ligne de commande

## Utilisation

### Jouer au Snake manuellement

```bash
python main.py --mode play
```

Utilisez les touches flèches pour contrôler le serpent.

### Générer un dataset et entraîner le modèle

```bash
python main.py --mode train --episodes 100 --epochs 50
```

Les modèles entraînés seront sauvegardés dans le dossier `models/`.

### Laisser le modèle jouer automatiquement

```bash
python main.py --mode simulate --model-path models/snake_model_best.pt
```

### Mode interactif (contrôles humains + prédictions du modèle)

```bash
python main.py --mode interactive --model-path models/snake_model_best.pt
```

## Options disponibles

```
usage: main.py [-h] [--mode {play,train,simulate,interactive}] [--size SIZE] [--episodes EPISODES] [--steps STEPS] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR] [--model-path MODEL_PATH]

Snake Neural Network Simulator

options:
  -h, --help            show this help message and exit
  --mode {play,train,simulate,interactive}
                        Mode to run: play (human controls), train (train model), simulate (model plays), interactive (human + model)
  --size SIZE           Size of the game board (default: 32*32)
  --episodes EPISODES   Number of episodes for training (default: 100)
  --steps STEPS         Maximum steps per episode (default: 500)
  --epochs EPOCHS       Number of training epochs (default: 50)
  --batch-size BATCH_SIZE
                        Batch size for training (default: 32)
  --lr LR               Learning rate (default: 0.001)
  --model-path MODEL_PATH
                        Path to saved model for simulation or interactive mode
```

## Fonctionnement

1. Le modèle est entrainé sur des données d'un vrai jeu Snake, où chaque exemple d'apprentissage est un triplet (état actuel, direction, état suivant).
2. Une fois entrainé, le modèle peut générer de nouveaux états de jeu en boucle, en utilisant sa propre sortie comme entrée pour l'étape suivante.
3. Dans le mode interactif, l'utilisateur fournit les directions et le modèle prédit comment le jeu évolue.

## Architecture détaillée

```
32*32 (état du jeu) → CNN Encodeur → [4096] latent → 
                                              ↓
                      [4] (direction) → Embedding → [64]
                                              ↓
                                          Attention
                                              ↓
                                            MLP
                                              ↓
                                     CNN Décodeur → 32*32 (nouvel état)
```

## Représentation du jeu

- **0** : Case vide
- **-1** : Serpent
- **1** : Nourriture
