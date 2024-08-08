using DelimitedFiles
using JuMP
using MathOptInterface
using CPLEX
using Dates
include("C:\\Users\\gabri\\.julia\\packages\\MultiObjectiveAlgorithms.jl-master\\src\\MultiObjectiveAlgorithms.jl") # Chemin mis à jour
import .MultiObjectiveAlgorithms as MOA
using DataStructures
using PyPlot


const MOI = MathOptInterface

# Lire les données
data1 = readdlm("instance/euclidA100.tsp")
data2 = readdlm("instance/euclidB100.tsp")

# Extraire les coordonnées des 5 premières villes pour chaque objectif
cities1 = data1[7:15, 2:3]
cities2 = data2[7:15, 2:3]

# Créer une matrice de distance pour chaque objectif
n = 9
dist_matrix1 = zeros(Int, n, n)
dist_matrix2 = zeros(Int, n, n)
for i in 1:n
    for j in 1:n
        dist_matrix1[i, j] = round(sqrt((cities1[i, 1] - cities1[j, 1])^2 + (cities1[i, 2] - cities1[j, 2])^2))
        dist_matrix2[i, j] = round(sqrt((cities2[i, 1] - cities2[j, 1])^2 + (cities2[i, 2] - cities2[j, 2])^2))
    end
end


model = Model()

# Création des variables binaires Xij
@variable(model, X[1:n, 1:n], Bin)


# Contraintes de flux de sortie et d'entrée
for i in 1:n
    @constraint(model, sum(X[i, :]) == 1)  
    @constraint(model, sum(X[:, i]) == 1)  
    @constraint(model, X[i, i] == 0)       
end


# Définir les objectifs
@expression(model, obj1, sum(dist_matrix1[i, j] * X[i, j] for i in 1:n for j in 1:n))
@expression(model, obj2, sum(dist_matrix2[i, j] * X[i, j] for i in 1:n for j in 1:n))
@objective(model, Min, [obj1, obj2])


# Créer un modèle pour l'algorithme de Branch-and-Bound multi-objectif
set_optimizer(model, () -> MOA.Optimizer(CPLEX.Optimizer))
# Définir l'algorithme Branch-and-Bound
set_attribute(model, MOA.Algorithm(), MOA.MultiObjectiveBranchBound())


# Optimiser le modèle
optimize!(model)

# Obtenir le nombre de solutions
n_solutions = MOI.get(model, MOI.ResultCount())
println("Solve complete. Found $n_solutions solution(s)")

# Extraire et afficher toutes les solutions
objective_values = []
path_matrices = []
for i in 1:n_solutions
    push!(objective_values, MOI.get(model, MOI.ObjectiveValue(i)))
    path_matrix = zeros(Int, n, n)
    for j in 1:n
        for k in 1:n
            path_matrix[j, k] = MOI.get(model, MOI.VariablePrimal(i), X[j, k])
        end
    end
    push!(path_matrices, path_matrix)
end

# Imprimer les points du front de Pareto dans le terminal
for (i, point) in enumerate(objective_values)
    println("Point $i : Objectif 1 = $(point[1]), Objectif 2 = $(point[2])")
    println("Matrice du chemin optimal : ")
    println(path_matrices[i])
end


# Créer un graphique avec toutes les solutions
scatter([x[1] for x in objective_values], [x[2] for x in objective_values])
title("Points supportés")
xlabel("Objectif 1")
ylabel("Objectif 2")

# Afficher le graphique
show()
