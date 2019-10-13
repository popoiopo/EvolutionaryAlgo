# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import matplotlib.pyplot as plt
from math import fabs, sqrt
import matplotlib as mpl
import multiprocess
import numpy as np
import glob, os
import datetime
import pickle
import dill
import time

#####################################################################################
################################## SCORE FUNCTIONS ##################################
################################### YOU MAY TOUCH ###################################
#####################################################################################


def fitnessGeneralist(fitness, playerLife, enemyLife, time):
    """Calculate general fitness over parDict['enemies'] amount of enemies

    Arguments:
        fitness {np.array} -- Array of parDict['enemies'] integer scores
        playerLife {np.array} -- Array of parDict['enemies'] remnants of player life
        enemyLife {np.array} -- Array of parDict['enemies'] remnants of enemy life
        time {[type]} -- Array of parDict['enemies'] integers of time it took to finish the game

    Returns:
        int -- General score over parDict['enemies'] scores
    """
    wins = (np.count_nonzero(np.array(enemyLife) == 0) / parDict["enemies"]) * 100
    plife = np.mean(playerLife)

    if parDict["verbose"]:
        print("GET GENERAL SCORE")
        print(fitness, playerLife, enemyLife, time)
        print("Generalist", wins)
        print(f"plife {plife}")

    return 0.7 * wins + 0.3 * plife - np.mean(np.log(time))


def fitness_single(playerLife, enemyLife, time):
    """Calculate fitness of a single run

    Arguments:
        playerLife {int} -- Remnant of players life
        enemyLife {int} -- Remnant (if any) of enemy life
        time {int} -- Time passed in being defeated or defeating an enemy

    Returns:
        int -- Fitness score of single run
    """
    return 0.9 * (100 - enemyLife) + 0.1 * playerLife - np.log(time)


#####################################################################################
################################## SCORE FUNCTIONS ##################################
################################### NO TOUCHY!!!! ###################################
#####################################################################################


def getScores(population):
    """Get scores for each individual in the population in a parallel manner

    Arguments:
        population {np.array} -- Array with parDict['popSize'] solutions of parDict['solutionSize'] length

    Returns:
        np.array -- Array of scores linked by index number to the population
    """
    manager = multiprocess.Manager()
    scores = manager.list()
    individualScores = manager.list()
    wins = manager.list()
    parameters = manager.list()
    jobs = []

    for x in population:
        p = multiprocess.Process(target=simulation, args=(x, scores, individualScores, wins, parameters))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return np.array(scores), np.array(individualScores), np.array(wins), np.array(parameters)


def simulation(x, scores, individualScores, wins, parameters):
    """Actual simulation that is parallelized. It runs parDict['enemies'] amount of randomly chosen enemies
       and applies a generalist score function on its output per enemy

    Arguments:
        x {np.array} -- Single solution of parDict['solutionSize']
        scores {multiprocessing.list()} -- List that is shared by the multiprocessing manager onto which results are appended

    Returns:
        np.array -- Set of scores (not actually used, but there as check)
    """
    np.random.seed()

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[2],
        playermode="ai",
        player_controller=player_controller(),
        enemymode="static",
        level=2,
        speed="fastest")

    fitness, playerLife, enemyLife, time = ([] for i in range(4))
    outputList = [fitness, playerLife, enemyLife, time]

    # default environment fitness is assumed for experiment
    env.player_controller.n_hidden = [parDict["N_hidden"]]
    env.state_to_log()  # checks environment state

    enemyFighters = np.random.choice(parDict["enemyrange"], parDict["enemies"])
    enemyFighters = list(range(1, parDict["enemies"]+1))
    for en in enemyFighters:
        env.update_parameter('enemies', [en])
        output = list(env.play(pcont=x))
        output[0] = fitness_single(*output[1:])

        if parDict["verbose"]:
            print("ENEMY", en)
            print(output)
        for lstData, lst in zip(output, outputList):
            lst.append(lstData)

    scores.append(fitnessGeneralist(*outputList))
    individualScores.append(fitness)
    wins.append(np.count_nonzero(np.array(enemyLife) == 0))
    parameters.append(x)
    return scores


#####################################################################################
################################### EA FUNCTIONS ####################################
################################### YOU MAY TOUCH ###################################
#####################################################################################


def eaMain():
    """Main function of the Genetic Algorithm, sets everything in motion

    Returns:
        np.arrays -- populations and scores for each generation
    """
    newPop = getPopulation(parDict["popSize"])
    scores, individualScores, wins, newPop = getScores(newPop)
    if parDict["verbose"]: print("*********** ENDSCORES **********\n", scores, individualScores, wins)

    print(f"Generation 0, Top score: {max(scores)}")

    populations, allScores, iScores, allWins = [list(newPop)], [list(scores)], [list(individualScores)], [list(wins)]

    for gens in range(parDict["eaGens"] - 1):
        eliteGen, newPop, eliteScores, newScores = getParents(scores, newPop)
        newPop = getNextGen(newPop, eliteGen)
        scores, individualScores, wins, newPop = getScores(newPop)

        print(f"Generation {gens+1}, Top score: {np.max(scores)}, Wins: {wins}")

        if parDict["verbose"]: print("*********** ENDSCORES **********\n", scores, individualScores, wins)

        populations.append(list(newPop))
        allScores.append(list(scores))
        iScores.append(list(individualScores))
        allWins.append(wins)

        # if gens+1 > parDict["breakDomain"] and checkStop(allScores): break

    return populations, allScores, iScores, allWins


def checkStop(allScores):
    """Give stopcondition when algorithm has converged
    (i.e. the best solutions have not changed for {parDict['breakDomain']} steps
    with a {parDict['minVar']} variance)

    Arguments:
        allScores {np.array} -- array of scores from every previous generation

    Returns:
        int -- 1 => break out of EA, 0 => proceed
    """
    lastN = allScores[-parDict["breakDomain"]:]
    if np.var(np.max(lastN, axis=1)) < parDict["minVar"]:
        return 1
    else:
        return 0


def getPopulation(size):
    """Function to get a solution population of specific size

    Arguments:
        size {int} -- Size of population

    Returns:
        np.array -- Genepool of {size} solutions
    """
    return np.random.uniform(-1, 1, (size, parDict["solutionSize"]))


def getParents(scores, population, distrib='linear', fit_offset=0.001):
    """Select parents for the next generation

    Arguments:
        scores {np.array} -- Scores of everyone in the parentpool population
        population {np.array} -- Array with parDict['popSize'] solutions of parDict['solutionSize'] length

    Keyword Arguments:
        distrib {str} -- [description] (default: {'linear'})
        fit_offset {float} -- [description] (default: {0.001})

    Returns:
        [type] -- [description]
    """
    scoreI = np.argsort(-scores)

    # get immigrants
    immigrantPars = getPopulation(parDict["numInflux"])
    immigrantScores, individualScores, wins, immigrantPars = getScores(immigrantPars)

    # select best solutions for elitism
    elitismPars = population[scoreI[:parDict["numElite"]]]
    elitismScores = scores[scoreI[:parDict["numElite"]]]

    if parDict["parentSelect"] == "random":
        randomI = np.random.permutation(range(
            len(population)))[:parDict["numRest"]]
        restPars = population[randomI]
        restScores = scores[randomI]

        return elitismPars, np.concatenate(
            [immigrantPars, elitismPars,
             restPars]), elitismScores, np.concatenate(
                 [immigrantScores, elitismScores, restScores])

    elif parDict["parentSelect"] == "fitness_proportional":
        if distrib == 'linear':
            fit_prop_scores = scores - min(scores) + fit_offset * min(
                max(scores), 0)
        elif distrib == 'quadratic':
            fit_prop_scores = (scores - min(scores))**2
            fit_prop_scores += fit_offset * min(max(fit_prop_scores), 0)
        elif distrib == 'root':
            fit_prop_scores = (scores - min(scores))**0.5
            fit_prop_scores += fit_offset * min(max(fit_prop_scores), 0)
        elif distrib == 'sigmoid':
            fit_prop_scores = [
                0.5 + math.erf(
                    (math.sqrt(math.pi) / 3) * (x - 0.5 * (max(scores)))) * 0.5
                for x in scores
            ]
            fit_prop_scores += fit_offset * min(max(fit_prop_scores), 0)
        else:
            print(
                'I dont know what probability distribution you want to use to fitness-proportionally draw parent candidates from.'
            )
        fit_prop_scores_norm = np.divide(fit_prop_scores, sum(fit_prop_scores))
        fp_parents_index = np.random.choice(
            range(len(population)),
            size=parDict["numRest"],
            replace=True,
            p=fit_prop_scores_norm)
        fp_parents = population[fp_parents_index]
        fp_scores = scores[fp_parents_index]

        return elitismPars, np.concatenate(
            [immigrantPars, elitismPars,
             fp_parents]), elitismScores, np.concatenate(
                 [immigrantScores, elitismScores, fp_scores])

    elif parDict["parentSelect"] == "ranking":
        ranking = [
            score for _, score in sorted(
                zip(scores, range(len(scores) - 1, -1, -1)))
        ]
        rank_props = list(map(distribution, ranking))
        rank_props_norm = np.divide(rank_props, sum(rank_props))
        rank_parents_index = np.random.choice(
            range(len(population)),
            size=parDict["numRest"],
            replace=True,
            p=rank_props_norm)
        rank_parents = population[rank_parents_index]
        rank_scores = scores[rank_parents_index]

        return elitismPars, np.concatenate(
            [immigrantPars, elitismPars,
             rank_parents]), elitismScores, np.concatenate(
                 [immigrantScores, elitismScores, rank_scores])
        # opening for different parent selection methods


def distribution(x):  # feel free to alter or expand
    """Distribution of TOBIAS NAAR KIJKEN ############################

    Returns:
        [type] -- [description]
    """
    return max(0, parDict["numRest"] - 1 * x)


def getNextGen(genepool, eliteGen, sim_con=False):
    """Umbrella function for creating the next generation from chosen genepool

    Arguments:
        genepool {np.array} -- Array with parDict['popSize'] solutions of parDict['solutionSize'] length
        eliteGen {np.array} -- Solutions that are advancing through elitism

    Keyword Arguments:
        sim_con {bool} -- [description] (default: {False})

    Returns:
        np.array -- Next generation genepool
    """
    nextGenCross = crossover(genepool, similarity_constraint=sim_con)
    nextG = mutations(nextGenCross)
    return np.concatenate([nextG, eliteGen])


def crossover(genepool,
              n_crosses=1,
              similarity_constraint=False,
              similarity_multiplier=2):
    """Crossover function dependent on parDict['crossoverType']

    Arguments:
        genepool {np.array} -- Array with parDict['popSize'] solutions of parDict['solutionSize'] length

    Keyword Arguments:
        n_crosses {int} -- [description] (default: {1})
        similarity_constraint {bool} -- [description] (default: {False})
        similarity_multiplier {int} -- [description] (default: {2})

    Returns:
        np.array -- Population with crossover
    """
    nextGen = []
    for i in range(0, len(genepool) - 1, 2):
        if similarity_constraint == True:
            similar = False
            while similar == False:
                p1 = genepool[np.random.choice(np.arange(len(genepool)))]
                p2 = genepool[np.random.choice(np.arange(len(genepool)))]
                if np.sqrt(sum(
                    (p1 - p2)**2)) / similarity_multiplier < np.random.uniform(
                        low=0, high=np.sqrt(4 * parDict["solutionSize"])):
                    similar = True
                else:
                    continue
        else:
            iParr = np.arange(len(genepool))
            np.random.shuffle(iParr)
            p1 = genepool[iParr[i]]
            p2 = genepool[iParr[i + 1]]

        if parDict["crossoverType"] == "randomRanged":
            iin = np.random.randint(0, high=len(p1), size=2)
            p1[min(iin):max(iin)], p2[min(iin):max(iin)] = p2[min(iin):max(
                iin)], p1[min(iin):max(iin)].copy()

        elif parDict["crossoverType"] == "n_crossover":
            # opening for different crossover methods
            crosses = np.random.randint(1, (len(p1)) - 2, size=n_crosses)
            kid1 = p1.copy()
            kid2 = p2.copy()
            for cross in crosses:
                kid1 = np.append(kid1[:cross], p2[cross:])
                kid2 = np.append(kid2[:cross], p1[cross:])
            p1 = kid1.copy()
            p2 = kid2.copy()

        elif parDict["crossoverType"] == "biased_crossover":
            biased_crosses = np.random.choice(
                range(len(p1)),
                size=n_crosses,
                p=parDict["biasvector"],
                replace=False)
            kid1 = p1.copy()
            kid2 = p2.copy()
            for cross in biased_crosses:
                kid1 = np.append(kid1[:cross], p2[cross:])
                kid2 = np.append(kid2[:cross], p1[cross:])
            p1 = kid1.copy()
            p2 = kid2.copy()

        else:
            pass

        nextGen.append(p1)
        nextGen.append(p2)
    return nextGen


def mutations(genepool):
    """Function that imposes mutations on the genepool. Method of mutation depends on parDict['mutationType']

    Arguments:
        genepool {np.array} -- Array with parDict['popSize'] solutions of parDict['solutionSize'] length

    Returns:
        np.array -- Mutated population
    """
    newGen = []
    if parDict["mutationType"] == "pointMutation":
        if parDict["mutationChanceType"] == "fixed":
            for child in genepool:
                if parDict["mutationChance"] > np.random.uniform():
                    child[np.random.choice(range(
                        len(child)))] += np.random.uniform(
                            *parDict["mutationDomain"])
                newGen.append(child)

        elif parDict["mutationChanceType"] == "random":
            for child in genepool:
                if np.random.uniform() > np.random.uniform():
                    child[np.random.choice(range(
                        len(child)))] += np.random.uniform(
                            *parDict["mutationDomain"])
                newGen.append(child)

    elif parDict["mutationType"] == "rangedMutation":
        if parDict["rangeType"] == "fixed":
            for child in genepool:
                for i in range(parDict["rangeSize"] * len(child)):
                    child[np.random.choice(range(
                        len(child)))] += np.random.uniform(
                            *parDict["mutationDomain"])
                newGen.append(child)

        elif parDict["rangeType"] == "random":
            for child in genepool:
                for i in range(np.random.uniform(0, len(child))):
                    child[np.random.choice(range(
                        len(child)))] += np.random.uniform(
                            *parDict["mutationDomain"])
                newGen.append(child)

    elif parDict["mutationType"] == "allMutation":
        for child in genepool:
            child += np.random.uniform(*parDict["mutationDomain"], len(child))
            newGen.append(child)

    elif parDict["mutationType"] == "MutateNodes":
        for child in genepool:
            if parDict["mutationChance"] > np.random.uniform():
                # do mutation
                mutatedbias = np.random.randint(parDict["N_hidden"] +
                                                parDict["num_outputs"])
                if np.random.choice(['bias', 'node']) == 'bias':
                    if mutatedbias > parDict["N_hidden"]:
                        child[parDict["endweight1"] + mutatedbias -
                              parDict["N_hidden"]] = min(
                                  max(
                                      child[parDict["endweight1"] + mutatedbias
                                            - parDict["N_hidden"]] +
                                      np.random.normal(0, 0.25), -1), 1)
                    else:
                        child[mutatedbias] = min(
                            max(child[mutatedbias] + np.random.normal(0, 0.25),
                                -1), 1)
                else:
                    if mutatedbias > parDict["N_hidden"]:
                        tempchild = child[parDict["endbias2"]:].reshape(
                            (parDict["N_hidden"], parDict["num_outputs"]))
                        tempchild[mutatedbias -
                                  parDict["N_hidden"], :] += np.random.normal(
                                      0, 0.25)
                        tempchild = tempchild.reshape(
                            1, parDict["N_hidden"] * parDict["num_outputs"])
                        child[parDict["endbias2"]:] = np.minimum(
                            np.maximum(tempchild,
                                       np.ones(len(tempchild)) * -1),
                            np.ones(len(tempchild)))
                    else:
                        tempchild = child[
                            parDict["N_hidden"]:parDict["endweight1"]].reshape(
                                (parDict["num_inputs"], parDict["N_hidden"]))
                        tempchild[:,
                                  parDict["mutatedbias"]] += np.random.normal(
                                      0, 0.25)
                        tempchild = tempchild.reshape(
                            1, parDict["num_inputs"] * parDict["N_hidden"])
                        child[parDict["N_hidden"]:
                              parDict["endweight1"]] = np.minimum(
                                  np.maximum(tempchild,
                                             np.ones(len(tempchild)) * -1),
                                  np.ones(len(tempchild)))
            newGen.append(child)

    else:
        pass
    return newGen

#####################################################################################
############################### HILLCLIMBER FUNCTIONS ###############################
################################## YES TOUCHY! :D ###################################
#####################################################################################


def hillClimber(iterations, filename):
    """Simple hillclimber implementation to estimate right EA parameters
    """
    bestParameters = {
        "eaGens": 3,
        "popSize": 10,
        "mutationDomain": (-0.05, 0.05),
        "numElite": 1,
        "numInflux": 1,
        "breakDomain": 1,
        "minVar": 1
    }
    hillScore = -100

    dObject = dataObject(parDict, filename)

    for i in range(iterations):
        parameters = perturbParameters(bestParameters)
        parDict.update(parameters)
        print(f"HillClimb iteration: {i}")

        # catch error if parameter values get negative
        try:
            dObject.addRep(*list(eaMain()))
            newScore = dObject.calcScore()
            if newScore > hillScore:
                print("Iteration", str(i), newScore)
                hillScore = newScore
                bestParameters = parameters
        except (TypeError, ValueError) as e:
            continue

    bestParameters = {"hillScore": hillScore, "parameters": parameters}
    pickle.dump(bestParameters, open(filename, "wb"))

    return bestParameters


def perturbParameters(parameters):
    """Function for perturbing EA parameters

    Arguments:
        parameters {dict} -- dictionary of all parameters that need to be optimized

    Returns:
        dict -- perturbed set of parameters
    """
    parameters['eaGens'] += np.random.randint(0, 5)
    parameters['popSize'] += np.random.randint(0, 5)
    parameters['mutationDomain'] += np.random.uniform(-0.01, 0.01, size=2)
    parameters['numElite'] += np.random.randint(0, 2)
    parameters['numInflux'] += np.random.randint(0, 2)
    parameters['breakDomain'] += np.random.randint(0, 2)
    parameters['minVar'] += np.random.uniform(0, 0.5)

    return parameters


#####################################################################################
#################################### DATA OBJECT ####################################
############################ NO TOUCH, UNLESS DELIBERATED ###########################
#####################################################################################


class simpleDataObject:
    """"""

    def __init__(self, populations, scores):
        self.populations = populations
        self.scores = scores


class dataObject:
    """"""

    def __init__(self, parameters, filename):
        self.populations = []
        self.scores = []
        self.iScores = []
        self.wins = []
        self.filename = filename
        self.parameters = parameters

    def addRep(self, population, scores, iScores, wins):
        self.populations.append(population)
        self.scores.append(scores)
        self.iScores.append(iScores)
        self.wins.append(wins)

    def load(self):
        f = open(self.filename, 'rb')
        tmp_dict = dill.load(f)
        f.close()
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)

    def save(self):
        f = open(self.filename, 'wb')
        dill.dump(self.__dict__, f, 2)
        f.close()

    def calcScore(self):
        return np.max(self.scores[-1])


#####################################################################################
############################## GLOBAL PARAMETER INIT ################################
############### YOU MAY TOUCH, ONLY VALUES THOUGH, UNLESS DELIBERATED ###############
#####################################################################################

# create folder to put experiment results in
experiment_name = "evoTest"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# create environment for some initialization purposes
env = Environment(
    experiment_name=experiment_name,
    enemies=[2],
    playermode="ai",
    player_controller=player_controller(),
    enemymode="static",
    level=2,
    speed="fastest")

# simulation specifics
parentSelectTypes = ["random", "fitness_proportional", "ranking"]
crossoverTypes = ["randomRanged", "n_crossover", "biased_crossover"]
mutationChanceTypes = ["fixed", "random"]
mutationTypes = [
    "pointMutation", "rangedMutation", "allMutation", "MutateNodes"
]
rangeTypes = ["fixed", "random"]

parDict = {
    "enemies": 3,
    "enemyrange": list(range(1, 9)),
    "eaGens": 10,
    "popSize": 20,
    "reps": 1,
    "N_hidden": 10,
    "num_inputs": env.get_num_sensors(),
    "num_outputs": 5,
    "parentSelect": parentSelectTypes[1],
    "crossoverType": crossoverTypes[2],
    "mutationChanceType": mutationChanceTypes[1],
    "mutationDomain": (-0.05, 0.05),
    "mutationType": mutationTypes[2],
    "rangeType": rangeTypes[0],
    "rangeSize": 0.1, # if ranged mutation, ratio of mutations of whole genome
    "numElite": 3,  # of population that gets retained into next generation
    "numInflux": 1,  # of population that is freshly immigrated
    "crossover_bias": 15,
    "breakDomain": 1, # number of generations check for stopcondition, 1 third of eaGens should be nice
    "minVar": 1, # variation threshold, if variation of last {breakDomain} generations < this -> break
    "verbose": False # true or false to see debugging prints
}

parDict["solutionSize"] = ((parDict["num_inputs"] + 1) * parDict["N_hidden"] +
                           (parDict["N_hidden"] + 1) * parDict["num_outputs"])
parDict["mutationChance"] = parDict["solutionSize"] * 1.75 / (
    parDict["popSize"] * np.sqrt(parDict["solutionSize"]))
parDict["numRest"] = parDict["popSize"] - (
    parDict["numElite"] * 2) - parDict["numInflux"]

parDict["endweight1"] = (parDict["num_inputs"] + 1) * parDict["N_hidden"]
parDict["endbias2"] = parDict["endweight1"] + parDict["num_outputs"]

parDict["biasvector"] = np.ones(
    ((parDict["num_inputs"] + 1) * parDict["N_hidden"] +
     (parDict["N_hidden"] + 1) * parDict["num_outputs"]))
parDict["biasvector"][:parDict["N_hidden"]] = parDict["biasvector"][
    parDict["num_inputs"] * parDict["N_hidden"] +
    parDict["N_hidden"]:parDict["num_inputs"] * parDict["N_hidden"] +
    parDict["N_hidden"] + parDict["num_outputs"]] = parDict["crossover_bias"]
parDict["biasvector"] = parDict["biasvector"] / sum(parDict["biasvector"])

#####################################################################################
############################## MAIN FLOW OF OPERATIONS ##############################
################################### NO TOUCHY!!!! ###################################
#####################################################################################

if __name__ == '__main__':
    # hillFilename = f"data/hillclimber_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
    # bestParameters = hillClimber(100, hillFilename)
    # print(bestParameters)

    # filename = f"data/bas/gens_{parDict['eaGens']}_psize_{parDict['popSize']}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
    filename = "test.pkl"
    dObject = dataObject(parDict, filename)

    for rep in range(parDict["reps"]):
        print(f"Repetition: {rep}")
        dObject.addRep(*list(eaMain()))

    print("######## WRITING AWAY DATA ########")
    testObject = dObject.scores
    dObject.save()

    print("######## TESTING WRITTEN DATA ########")
    test = dataObject("", filename)
    test.load()

    assert np.array_equal(test.scores, testObject), ("ERROR with pickling occured", test.scores, "\n", testObject)
    print("Everything went right!")