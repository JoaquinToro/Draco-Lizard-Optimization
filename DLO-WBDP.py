import numpy as np
import math, random

def levy_flight(size, alpha=1.5, beta=0.5):
    sigma = (math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / (
                math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))) ** (1 / alpha)
    u = np.random.normal(0, sigma, size)
    v = np.random.normal(0, 1, size)
    r = u / (abs(v) ** (1 / alpha))
    x = np.random.normal(0, (1 / beta), size)
    return x * r

def DLO(SearchAgents_no, Tmax, ub, lb, dim, fobj):
    Convergence_curve = np.ones(Tmax) * np.inf
    Global_Best_position = np.ones(dim)
    Global_Best_fitness = np.inf

    Individual_Best_Fitness = np.ones(SearchAgents_no) * np.inf
    # Initialization
    Individual_Best_Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Individual_Best_Positions[:, i] = np.random.rand(SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    for i in range(SearchAgents_no):
        L = Individual_Best_Positions[i, :].copy()
        Individual_Best_Fitness[i] = fobj(L)
        if Individual_Best_Fitness[i] < Global_Best_fitness:
            Global_Best_fitness = Individual_Best_Fitness[i].copy()
            Global_Best_position = Individual_Best_Positions[i, :].copy()

    Positions_NEW = Individual_Best_Positions.copy()
    Fitness_NEW = Individual_Best_Fitness.copy()

    # Main loop
    for t in range(Tmax):
        for i in range(SearchAgents_no):
            # Exploration
            if t < Tmax/2:
                I = round(1 + np.random.random())
                k = np.random.choice(range(SearchAgents_no))
                j = np.random.choice(range(SearchAgents_no))
                P = Individual_Best_Positions[k, :].copy()
                F_P = Individual_Best_Fitness[k]

                if t<Tmax*0.1:
                    Sigma = np.array([random.choice([1, 1, 1, 1]) for _ in range(dim)])
                elif t < Tmax * 0.2:
                    Sigma = np.array([random.choice([0, 1, 1, 1]) for _ in range(dim)])
                elif t<Tmax * 0.2:
                    Sigma = np.array([random.choice([0, 0, 1, 1]) for _ in range(dim)])
                elif t< Tmax* 0.4:
                    Sigma = np.array([random.choice([0, 0, 0, 1]) for _ in range(dim)])
                else:
                    Sigma = np.array([random.choice([0, 0, 0, 0]) for _ in range(dim)])
                if Individual_Best_Fitness[i] > F_P:
                    print(f'x[{t+1}] = {Individual_Best_Positions[i, :].copy()} + {np.random.rand(dim)} * ({P} - {I} * {Individual_Best_Positions[i, :].copy()}) + {Sigma} * {np.random.rand(dim)} * ({P} - {Individual_Best_Positions[j, :].copy()})')
                    Positions_NEW[i, :] = Individual_Best_Positions[i, :].copy() + np.random.rand(dim) * (
                            P - I * Individual_Best_Positions[i, :].copy()) + Sigma * np.random.rand(dim) * (
                            P - Individual_Best_Positions[j, :].copy())
                else:
                    print(f'x[{t+1}] = {Individual_Best_Positions[i, :].copy()} + {np.random.rand(dim)} * ({Individual_Best_Positions[i, :].copy()} - {P}) + {Sigma} * {np.random.rand(dim)} * ({P} - {Individual_Best_Positions[j, :].copy()})')
                    Positions_NEW[i, :] = Individual_Best_Positions[i, :].copy() + np.random.rand(dim) * (
                            Individual_Best_Positions[i, :].copy() - P) + Sigma * np.random.rand(dim) * (
                            P - Individual_Best_Positions[j, :].copy() )
            # Exploitation
            else:
                
                SearchAgents_no_list = list(range(SearchAgents_no))
                SearchAgents_no_list.remove(i)
                m = random.sample(SearchAgents_no_list, 1)
                selected_searchAgent = Individual_Best_Positions[m, :].copy()

                p = np.random.random()

                if p < 0.2:
                    print(f"x[{t+1}] = {Global_Best_position} + {levy_flight(dim)} * ( {Global_Best_position} - {selected_searchAgent})")
                    Positions_NEW[i, :] = (Global_Best_position + levy_flight(dim) *
                                        (Global_Best_position - selected_searchAgent))
                else:
                    print(f"x[{t+1}] = {Global_Best_position} + {np.random.normal(loc=0.0, scale=10)} * ( {(1-t/Tmax)} * {(Individual_Best_Positions[i, :] - selected_searchAgent)})")
                    Positions_NEW[i, :] = (Global_Best_position + np.random.normal(loc=0.0, scale=10) * (1-t/Tmax) *
                                        (Individual_Best_Positions[i, :] - selected_searchAgent))

            for d in range(dim):
                if Positions_NEW[i, d] > ub[d]:
                    Positions_NEW[i, d] = lb[d] + np.random.random() * (ub[d] - lb[d])
                elif Positions_NEW[i, d] < lb[d]:
                    Positions_NEW[i, d] = lb[d] + np.random.random() * (ub[d] - lb[d])

            L = Positions_NEW[i, :].copy()
            Fitness_NEW[i] = fobj(L)

            # PARA MINIMIZACION
            if Fitness_NEW[i] < Individual_Best_Fitness[i]:
                Individual_Best_Fitness[i] = Fitness_NEW[i]
                Individual_Best_Positions[i, :] = Positions_NEW[i].copy()

                if Individual_Best_Fitness[i] < Global_Best_fitness:
                    Global_Best_fitness = Individual_Best_Fitness[i]
                    Global_Best_position = Individual_Best_Positions[i, :].copy()

        Convergence_curve[t] = Global_Best_fitness
    return Global_Best_fitness, Global_Best_position, Convergence_curve

# Funcion objetivo (nose si es esta pero e)
def funcion_objetivo(x):
    h, l, t, b = x
    return 1.10471*h**2*l + 0.04811*t*b*(14.0+l)

# No entendi las restricciones asi q puse cualquier wea
def comprobar_restricciones(x):
    h, l, t, b = x
    cumple = True

    if l < h and t > b:
        cumple = False

    return cumple

# Comprobar si cumple las restricciones, si las cumple, calcular el fitness
def welded_beam_fitness(x):
    if comprobar_restricciones(x):
        return funcion_objetivo(x)
    else:
        return np.inf

#Limites
lb = [0.125,0.1,0.1,0.125]
ub = [5,10,10,5]
dim = 4 

# Ejecutar DLO
best_fitness, best_position, convergence_curve = DLO(
    SearchAgents_no=10,
    Tmax=100,
    ub=ub,
    lb=lb,
    dim=dim,
    fobj=welded_beam_fitness
)

#Imprimir resultados
print("Costo mínimo encontrado:", best_fitness)
print("Mejor solución encontrada:", best_position)
