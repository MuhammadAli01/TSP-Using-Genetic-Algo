import random
from PIL import Image, ImageDraw, ImageChops
import numpy as np


# load target image and resize
IMG = Image.open('test.jpg').convert('RGB')
IMG_SIZE = (300, 300)
TARGET = IMG.resize(IMG_SIZE)

MUTATION_CHANCE = 0.01
ADD_GENE_CHANCE = 0.3
REM_GENE_CHANCE = 0.2

INITIAL_GENES = 50

NO_OF_SIDES = 6


class Gene:
    def __init__(self):
        self.posList = [(random.randint(0, IMG_SIZE[0]), random.randint(0, IMG_SIZE[1])) for i in range(NO_OF_SIDES)]
        self.color = {'r': random.randint(0, 255),
                      'g': random.randint(0, 255),
                      'b': random.randint(0, 255),
                      'a': 128}

    def mutate(self):
        types = ["corner", "pos", "color"]
        mut_type = random.choice(types)

        # Randomly change position of one corner
        if mut_type == "corner":
            rand_ind = random.randint(0, len(self.posList))
            self.posList[rand_ind] = random.randint(0, IMG_SIZE + 1)

        elif mut_type == "pos":
            self.posList = [(random.randint(0, IMG_SIZE[0]),
                             random.randint(0, IMG_SIZE[1])) for i in range(NO_OF_SIDES)]

        elif mut_type == "color":
            self.color = {'r': random.randint(0, 255),
                          'g': random.randint(0, 255),
                          'b': random.randint(0, 255),
                          'a': 128}


class Chromosome:
    def __init__(self, parentA=None, parentB=None):
        self.genes = [Gene() for i in range(INITIAL_GENES)]

    def mutate(self):
        for g in self.genes:
            if MUTATION_CHANCE < random.random():
                g.mutate()

        if ADD_GENE_CHANCE < random.random():
            self.genes.append(Gene())
        if len(self.genes) > 0 and REM_GENE_CHANCE < random.random():
            self.genes.remove(random.choice(self.genes))

    def drawImage(self):
        image = Image.new("RGB", IMG_SIZE, (255, 255, 255))
        canvas = ImageDraw.Draw(image, 'RGBA')

        for gene in self.genes:
            color = (gene.color['r'], gene.color['g'], gene.color['b'], gene.color['a'])
            canvas.polygon([gene.posList], fill=color)

        return image


def findFitness(img_1, img_2):
    """
    :param img_1:
    :param img_2:
    :return: % difference between the two images
    Converts both images to numpy arrays and finds their percentage difference
    """
    i1 = np.array(img_1, np.int16)
    i2 = np.array(img_2, np.int16)
    dif = np.sum(np.abs(i1-i2))
    return (dif / 255.0 * 100) / i1.size

def crossover(parent_a, parent_b):
    crossPoint = random.randint(0, len(parent_a.genes)-1)
    child = Chromosome()
    child.genes = parent_a.genes[:crossPoint] + parent_b.genes[crossPoint:]
    return child

