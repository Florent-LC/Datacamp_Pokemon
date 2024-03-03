@@ -1,2 +1,36 @@
# Datacamp - Pokemon Classification Challenge

The team project for Datacamp 2023-2024
# RAMP starting kit on Pokemon Type Classification

Authors:
Kyllian Asselin De Beauville, Yann Choho, Julia Cuvelier, Théo Gerard, Thierry-Séphine Goma-Legernard, Florent Le Clerc

The Pokemon universe is vast and diverse, with creatures of various shapes, sizes, and types scattered across the digital landscape. Each Pokemon has specific types that determine its abilities, strengths and weaknesses.
Classifying Pokemon by their type is not only crucial for trainers to prepare for battles and to strategize their gameplay but also for understanding the intricate relationships between different Pokemon species.

In the world of Pokemon, types are not just visual characteristics but also encapsulate the essence of a Pokemon's abilities and lore. This provides a rich dataset for applying not only Computer Vision but also Natural Language Processing (NLP) techniques. The names of Pokemon, for instance, often hold clues to their types and elemental attributes.

The goal of this RAMP is to predict the type of Pokemon based on scraped images from the Poképedia website [Poképedia](https://www.pokepedia.fr/Liste_des_Pokémon_dans_l'ordre_du_Pokédex_National) and textual data such as their names. Participants are encouraged to explore and combine Computer Vision and NLP techniques to analyze the images and text data to predict Pokemon types accurately.

#### Set up

Open a terminal and

0. Install all requirements present in the ```requirements.txt```

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install ramp-workflow
  ```

2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](starting_kit.ipynb).
To test the starting-kit, run

```
ramp-test --quick-test
```

#### Help

If you encounter any issues or have questions, consult the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for additional information and guidance on the [RAMP](https://ramp.studio) ecosystem.