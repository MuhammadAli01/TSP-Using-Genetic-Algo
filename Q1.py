import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Globals
popSize = 50
noOfOffspring = 20
noOfGeneration = 3000
mutationRate = 0.5


class City:
    def __init__(self, cityId, x, y):
        self.cityId = cityId
        self.x = x
        self.y = y

    def __repr__(self):
        return str(self.cityId)

    def findDist(self, cityB):
        """
        Calculates distance between 2 cities using Pythagoras theorem
        """
        return np.sqrt((cityB.x - self.x)**2 + ((cityB.y - self.y)**2))


def initializeCities(filename):
    """
    Reads from file and returns list of cities. Skips first 7 lines and the last line of file.
    """
    cities = []
    with open(filename) as file:
        for line in file.readlines()[7:-1]:
            strId, strX, strY = line.split(" ")
            cities.append(City(int(strId), float(strX), float(strY)))
    return cities


def initializePopulation(citiesList, popSize):
    """
    Returns array of randomly initialized population
    """
    # return [random.shuffle(citiesList.copy()) for i in range(popSize)]
    population = []
    for i in range(popSize):
        population.append(citiesList.copy())
        random.shuffle(population[i])
    return population


def findFitness(route):
    return 1 / sum((route[i].findDist(route[(i+1) % len(route)])) for i in range(len(route)))
    # totalDist = 0
    # for i in range(len(route)):
    #     totalDist += route[i].findDist(route[(i+1) % len(route)])
    # return 1 / totalDist


def selection(df, parSelSch):
    """
    :param df: World datframe
    :param parSelSch: Selection scheme. random, FPS, rank, tournament, truncation
    :param tourSize:
    :return:
    """
    if parSelSch.lower() == "fps":
        return df.sample(n=2, weights="Fitness").index.values
    elif 'rank' in parSelSch.lower():
        temp_df = df.sort_values('Fitness')
        weightsLst = range(1, len(temp_df.index) + 1)
        return temp_df.sample(n=2, weights=weightsLst).index.values
    elif 'tourn' in parSelSch.lower():
        parentLst = []
        for i in range(2):
            temp_df = df.sample(n=2)
            parentLst.append(temp_df['Fitness'].idxmax())
        return parentLst
    elif parSelSch.lower() == 'random':
        return df.sample(n=2).index.values
    elif 'trunc' in parSelSch.lower():
        temp_df = df.sort_values('Fitness', ascending=False)
        return temp_df[:5].sample(n=2).index.values
    else:
        print("Invalid parent selection scheme")




def breed(df, parIndices, mutationRate):
    """
    Given world dataframe, indexes of two parents, it returns their child.
    Performs crossover and mutation.
    """
    noOfCities = len(df.iloc[0].Route)
    routeA, routeB = df.iloc[parIndices[0]].Route, df.iloc[parIndices[1]].Route
    start = random.randint(0, noOfCities - 1)  # Crossover
    end = random.randint(start, noOfCities - 1)
    child = routeA[start: end]
    child += [city for city in routeB if city not in set(child)]
    if random.random() < mutationRate:
        randIndex = random.randint(0, noOfCities - 1)
        temp = child[randIndex]
        child[randIndex] = child[(randIndex+1) % noOfCities]
        child[(randIndex+1) % noOfCities] = temp
    return child


def createNewPopulation(df, noOfOffspring, parSelSch, mutationRate):
    """
    Returns new population, including offspring
    :param df: World dataframe
    :param noOfOffspring:
    :return df containing new population (with offspring):
    """
    for i in range(noOfOffspring):
        child = breed(df, selection(df, parSelSch), mutationRate)
        df = df.append({'Route': child, 'Fitness': findFitness(child)}, ignore_index=True)
    return df


def killFromPop(df, popSize, surSelSch):
    """
    Kills from the new population by sorting and killing least fit members.
    :param df:
    :param popSize:
    :return: df of new generation, which has popSize number of entries.
    """
    if surSelSch.lower() == "fps":
        new_pop = df.sample(n=popSize, weights="Fitness")
    elif 'rank' in surSelSch.lower():
        df.sort_values('Fitness', inplace=True)
        new_pop = df.sample(n=popSize, weights=range(1, len(df.index) + 1))
    elif "trunc" in surSelSch.lower():
        df.sort_values('Fitness', inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)  # REMOVE LINE
        new_pop = df[:popSize]
    elif surSelSch.lower() == 'random':
        new_pop = df.sample(n=popSize)
    elif 'tourn' in surSelSch.lower():
        for i in range(len(df.index) - popSize):
            temp_df = df.sample(n=2)
            # print(f"Dropping row {temp_df['Fitness'].idxmin()}")
            df.drop(temp_df['Fitness'].idxmin(), inplace=True)
        new_pop = df
    else:
        print("Invalid survivor selection scheme.")
    new_pop.reset_index(drop=True, inplace=True)
    return new_pop


def getBSF(citiesList, parSelScheme, surSelScheme, noOfIterations):
    bsf_df = pd.DataFrame(index=range(1, noOfGeneration + 2))
    asf_df = pd.DataFrame(index=range(1, noOfGeneration + 2))
    for it in range(noOfIterations + 1):
        world_df = pd.DataFrame({'Route': initializePopulation(citiesList, popSize)})
        world_df['Fitness'] = world_df['Route'].map(findFitness)
        bsfLst, asfLst = [], []
        for genNo in range(noOfGeneration + 1):
            world_df = createNewPopulation(world_df, noOfOffspring, parSelScheme, mutationRate)
            world_df = killFromPop(world_df, popSize, surSelScheme)
            bsfLst.append(1 / (world_df['Fitness'].max()))
            asfLst.append(1 / (world_df['Fitness'].mean()))
            if genNo % 100 == 0:
                print(it, genNo)
        bsf_df[f'Run #{it + 1} BSF'] = bsfLst
        asf_df[f'Run #{it + 1} ASF'] = asfLst
    bsf_df['Average BSF'] = bsf_df.mean(axis=1)
    asf_df['Average ASF'] = asf_df.mean(axis=1)
    print(bsf_df.iloc[::10])
    print(asf_df)
    filename = '3000gen/' + parSelScheme + '-' + surSelScheme + '.csv'
    with open(filename, 'w+') as f:
        bsf_df.iloc[::10].to_csv(f)
    with open(filename, 'a+') as f:
        asf_df.iloc[::10].to_csv(f)
    return [bsf_df, asf_df]


def main():
    citiesList = initializeCities("qa194.tsp")
    # selSchemes = [("FPS", "FPS"), ("FPS", "Random"), ("Tournament", "Truncation"), ("Truncation", "Truncation")]
                  # ('Random', "Random"), ('FPS', 'Tournament'), ('Rank', 'Tournament'), ('Tournament', 'Tournament')]
    selSchemes = [('Random', "Random")]
    for parSelScheme, surSelScheme in selSchemes:
        print(f"parSelScheme: {parSelScheme}, surSelScheme: {surSelScheme}")
        dfLst = getBSF(citiesList, parSelScheme, surSelScheme, 3)
        ax = dfLst[0].plot.line(y='Average BSF', title=parSelScheme + '-' + surSelScheme, use_index=True, color='red')
        dfLst[1].plot.line(y='Average ASF', use_index=True, color='blue', ax=ax)
        plt.savefig('3000gen/' + parSelScheme + '-' + surSelScheme + '.png')

main()




# Initialize World
citiesList = initializeCities("qa194.tsp")
# world_df = pd.DataFrame({'Route': initializePopulation(citiesList, popSize)})
# world_df['Fitness'] = world_df['Route'].map(findFitness)

# getBSF(citiesList, 'tourn', 'tourn', 3)

# lst = getBSF(citiesList, 'tourn', 'tourn', 3)
#
# ax = lst[0].plot.line(y='Average BSF', use_index=True, color='red')
# lst[1].plot.line(y='Average ASF', use_index=True, color='blue', ax=ax)
# plt.show()
# plt.savefig('tour-tour.png')

# world_df = createNewPopulation(world_df, noOfOffspring, "rank", mutationRate)
# world_df = killFromPop(world_df, popSize, 'tourn')
# print("World_df: ")
# print(world_df)

# Code to test selection
# count_dict = {rank: 0 for rank in range(30)}
# for i in range(1000):
#     for j in selection(world_df, "rank"):
#         count_dict[j] += 1
# print(count_dict)

# print(selection(world_df, "trunc"))


# with open('initial_results.txt', 'w+') as fh:
#     fh.write(f"popSize = {popSize}, noOfOffspring = {noOfOffspring}, noOfGeneration = {noOfGeneration}, "
#              f"mutationRate = {mutationRate}\n")
#     selSchemes = [("FPS", "FPS"),  ("FPS", "random"), ("tourn", "trunc"), ('random', "random"), ('FPS', 'tourn'),
#                   ('rank', 'tourn'), ('tourn', 'tourn')]
#     for parSelScheme, surSelScheme in selSchemes:
#             print(f"PARENT SEL SCHEME: {parSelScheme}, SURVIVOR SELECTION SCHEME: {surSelScheme}")
#             fh.write(f"PARENT SEL SCHEME: {parSelScheme}, SURVIVOR SELECTION SCHEME: {surSelScheme}\n\n")
#             world_df = pd.DataFrame({'Route': initializePopulation(citiesList, popSize)})
#             world_df['Fitness'] = world_df['Route'].map(findFitness)
#             for genNo in range(noOfGeneration + 1):
#                 world_df = createNewPopulation(world_df, noOfOffspring, parSelScheme, mutationRate)
#                 world_df = killFromPop(world_df, popSize, surSelScheme)
#                 if genNo % (noOfGeneration/5) == 0:
#                     fh.write(f"Generation number: {genNo}. Best dist: {1 / (world_df['Fitness'].max())}\n")
#             fh.write("-" * 30)
#             fh.write("\n" * 3)

# with open('tourn_trunc.txt', 'w+') as fh:
#     fh.write(f"popSize = {popSize}, noOfOffspring = {noOfOffspring}, noOfGeneration = {noOfGeneration}, "
#              f"mutationRate = {mutationRate}\n")
#     parSelScheme = 'tourn'
#     surSelScheme = 'trunc'
#     for popSize in [30, 50, 100, 500, 1000]:
#         for noOfOffspring in [10, 20, 50, 100]:
#             if popSize > noOfOffspring:
#                 print(f"popSize: {popSize}, noOfOffSpring: {noOfOffspring}")
#                 fh.write(f"popSize: {popSize}, noOfOffSpring: {noOfOffspring}\n\n")
#                 world_df = pd.DataFrame({'Route': initializePopulation(citiesList, popSize)})
#                 world_df['Fitness'] = world_df['Route'].map(findFitness)
#                 for genNo in range(noOfGeneration + 1):
#                     world_df = createNewPopulation(world_df, noOfOffspring, parSelScheme, mutationRate)
#                     world_df = killFromPop(world_df, popSize, surSelScheme)
#                     if genNo % (noOfGeneration/5) == 0:
#                         fh.write(f"Generation number: {genNo}. Best dist: {1 / (world_df['Fitness'].max())}\n")
#                 fh.write("-" * 30)
#                 fh.write("\n" * 3)

# with open('initial_results.txt', 'w+') as fh:
#     fh.write(f"popSize = {popSize}, noOfOffspring = {noOfOffspring}, noOfGeneration = {noOfGeneration}\n")
#     # selSchemes = [("FPS", "FPS"),  ("FPS", "random"), ("tourn", "trunc"), ('random', "random"), ('FPS', 'tourn'),
#     #               ('rank', 'tourn'), ('tourn', 'tourn')]
#     selSchemes = [("FPS", "FPS"), ("tourn", "trunc"), ('FPS', 'tourn'), ('rank', 'tourn'), ('tourn', 'tourn')]
#     for parSelScheme, surSelScheme in selSchemes:
#             print(f"PARENT SEL SCHEME: {parSelScheme}, SURVIVOR SELECTION SCHEME: {surSelScheme}")
#             fh.write(f"PARENT SEL SCHEME: {parSelScheme}, SURVIVOR SELECTION SCHEME: {surSelScheme}\n\n")
#             for mutationRate in [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]:
#                 world_df = pd.DataFrame({'Route': initializePopulation(citiesList, popSize)})
#                 world_df['Fitness'] = world_df['Route'].map(findFitness)
#                 for genNo in range(noOfGeneration + 1):
#                     world_df = createNewPopulation(world_df, noOfOffspring, parSelScheme, mutationRate)
#                     world_df = killFromPop(world_df, popSize, surSelScheme)
#                     if genNo % (noOfGeneration/5) == 0:
#                         fh.write(f"Generation number: {genNo}. Best dist: {1 / (world_df['Fitness'].max())}\n")
#                 fh.write("-" * 30)
#                 fh.write("\n" * 3)


# for i in range(2000):
#     world_df = createNewPopulation(world_df, noOfOffspring, 'tourn', mutationRate)
#     world_df = killFromPop(world_df, popSize, 'tourn')
#     print(f"Generation number: {i}. Best dist: {1/(world_df['Fitness'].max())}")

