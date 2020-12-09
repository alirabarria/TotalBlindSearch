# -*- coding: latin-1 -*-
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
class Manage_plots:
    def __init__(self, max = 100, dpi = 400, use_max = False):
        self.max = max
        self.figs = []
        self.fig_names = []
        self.dpi = dpi
        self.use_max = use_max

    def add_plot(self, fig, fig_name):
        self.figs.append(fig)
        self.fig_names.append(fig_name)
        if len(self.figs) > self.max and self.use_max:
            self.save_plots()

    def save_plots(self, clean_figs = False, draw = True):

        for j in range(len(self.figs)):
            name = self.fig_names[j]
            fig = self.figs[j]
            if draw:
                canvas = FigureCanvas(fig)
                canvas.print_figure(name, dpi = self.dpi)
            else:
                self.figs[j].savefig(name, dpi = self.dpi)
        if clean_figs:
            for j in range(len(self.figs)):
                self.figs[j].clf()
            self.fig_names = []
            self.figs = []
            plt.close('all')

    def show_plots(self, index = -1):
        for j in range(len(self.figs)):
            if j == index:
                pass


