import os
import numpy as np
import errno
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch

class Registro:
    def __init__(self, nombre_modelo, nombre_datos):
        self.nombre_modelo = nombre_modelo
        self.nombre_datos = nombre_datos
        self.comment = '{}_{}'.format(nombre_modelo, nombre_datos)
        self.subdirectorio_datos = '{}/{}'.format(nombre_modelo, nombre_datos)
        self.writer = SummaryWriter(comment=self.comment)

    def guardar_img(self, fig, epoca, lote, comment=''):
        directorio = './data/imagenes/{}'.format(self.subdirectorio_datos)
        Registro._make_dir(directorio)
        fig.savefig('{}/{}_epoca_{}_lote_{}.png'.format(directorio, comment, epoca, lote))
        
    def guardar_img_torch(self, horizontal_grid, grid, epoca, lote, plot_horizontal=True):
        directorio = './data/imagenes/{}'.format(self.subdirectorio_datos)
        Registro._make_dir(directorio)
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(horizontal_grid.numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self.guardar_img(fig, epoca, lote, 'horizontal')
        plt.close()
        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
        plt.axis('off')
        self.guardar_img(fig, epoca, lote)
        plt.close()

    def registrar(self, error_D, error_G, epoca, lote, n_lotes):
        if isinstance(error_D, torch.autograd.Variable):
            error_D = error_D.data.cpu().numpy()
        if isinstance(error_G, torch.autograd.Variable):
            error_G = error_G.data.cpu().numpy()
        paso = Registro._paso(epoca, lote, n_lotes)
        self.writer.add_scalar(
            '{}/error_D'.format(self.comment), error_D, paso)
        self.writer.add_scalar(
            '{}/error_G'.format(self.comment), error_G, paso)

    def registrar_imagenes(self, imagenes, n_imagenes, epoca, lote, n_lotes, formato='NCHW', normalize=True):
        if type(imagenes) == np.ndarray:
            imagenes = torch.from_numpy(imagenes)
        if formato=='NHWC':
            imagenes = imagenes.transpose(1,3)
        paso = Registro._paso(epoca, lote, n_lotes)
        img_name = '{}/imagenes{}'.format(self.comment, '')
        horizontal_grid = vutils.make_grid(imagenes, normalize=normalize, scale_each=True)
        n_filas = int(np.sqrt(n_imagenes))
        grid = vutils.make_grid(imagenes, nrow=n_filas, normalize=True, scale_each=True)
        self.writer.add_image(img_name, horizontal_grid, paso)
        self.guardar_img_torch(horizontal_grid, grid, epoca, lote)

    def mostrar_estado(self, epoca, n_epocas, lote, n_lotes, error_D, error_G, pred_X, pred_G):
        if isinstance(error_D, torch.autograd.Variable):
            error_D = error_D.data.cpu().numpy()
        if isinstance(error_G, torch.autograd.Variable):
            error_G = error_G.data.cpu().numpy()
        if isinstance(pred_X, torch.autograd.Variable):
            pred_X = pred_X.data
        if isinstance(pred_G, torch.autograd.Variable):
            pred_G = pred_G.data
        print('Epoca: [{}/{}], Numero de Lote: [{}/{}]'.format( epoca, n_epocas, lote, n_lotes))
        print('D(G(z)): {:.4f}, D(G(x)): {:.4f}'.format(pred_G.mean(), pred_X.mean()))
        print('Perdida Discriminador: {:.4f}, Perdida Generador: {:.4f}'.format(error_D, error_G))
        

    def guardar_modelos(self, generador, discriminador, epoca):
        directorio = './data/modelos/{}'.format(self.subdirectorio_datos)
        Registro._make_dir(directorio)
        torch.save(generador.state_dict(), '{}/G_epoca_{}'.format(directorio, epoca))
        torch.save(discriminador.state_dict(), '{}/D_epoca_{}'.format(directorio, epoca))

    def close(self):
        self.writer.close()


    @staticmethod
    def _paso(epoca, lote, n_lotes):
        return epoca * n_lotes + lote

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise