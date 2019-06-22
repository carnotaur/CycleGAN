from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt 


class MakeThanks(object):

    
    @staticmethod
    def show_thanks():
        print('Спасибо за просмотр!!!')

        style = np.random.choice(['seaborn', 'ggplot'])
        plt.style.use(style)

        degree = np.random.rand()

        plt.figure(figsize=(15, 10))

        plt.text(0.6, 0.7, "Спасибо за просмотр ежже", size=50, rotation=degree,
         ha="center", va="center",
         bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
        plt.axis('off')
        plt.show()


def load_weights(G_x, G_y, D_x, D_y, 
                 optimizer_G, optimizer_D,
                 epoch: int, models_dir: str = 'models'):

    models_dir = Path(models_dir)
    path = models_dir.joinpath('gan_{}'.format(epoch))
    checkpoint = torch.load(path)
    
    G_x.load_state_dict(checkpoint['genX_state_dict'])
    G_y.load_state_dict(checkpoint['genY_state_dict'])
    D_x.load_state_dict(checkpoint['disX_state_dict'])
    D_y.load_state_dict(checkpoint['disY_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])