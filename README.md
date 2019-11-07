# player-networks
Code and description for the computation of Player Similarity Networks.

A extensive documentation of the method can be found in the PDF file.

# Code
Run `pn_main.py` and see the instructions at the beginning. I am working on adding more features and docu.

# Interpretation of the network
Players that are connected by an edge are similar in playing style. The edge color represents the strength of the connection. The whiter an edge, the more similar are the players.\\
Node color encodes a players' cluster. Purple nodes are noise (see below for details), i.e. players with few similar players in the sample.
