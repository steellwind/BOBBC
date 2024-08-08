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
cities1 = data1[7:12, 2:3]
cities2 = data2[7:12, 2:3]

# Créer une matrice de distance pour chaque objectif
n = 6
dist_matrix1 = zeros(Int, n, n)
dist_matrix2 = zeros(Int, n, n)
for i in 1:n
    for j in 1:n
        dist_matrix1[i, j] = round(sqrt((cities1[i, 1] - cities1[j, 1])^2 + (cities1[i, 2] - cities1[j, 2])^2))
        dist_matrix2[i, j] = round(sqrt((cities2[i, 1] - cities2[j, 1])^2 + (cities2[i, 2] - cities2[j, 2])^2))
    end
end


# Créer un modèle pour l'algorithme de Branch-and-Bound multi-objectif
model = Model(() -> MOA.Optimizer(CPLEX.Optimizer))

# Définir l'algorithme Branch-and-Bound
MOI.set(model, MOA.Algorithm(), MOA.MultiObjectiveBranchBound())

# Création des variables binaires Xij
@variable(model, X[1:n, 1:n], Bin)

# Création des variables de flots z_k_ij pour chaque k
@variable(model, z[2:n, 1:n, 1:n], Bin)


# Contraintes de sous-tours : chaque ville doit être visitée exactement une fois
for i in 1:n
    @constraint(model, sum(X[i, j] for j in 1:n if j != i) == 1)
    @constraint(model, sum(X[j, i] for j in 1:n if j != i) == 1)
    @constraint(model, X[i, i] == 0)
end

# Contraintes de flux
for k in 2:n
    @constraint(model, sum(z[k, 1, j] for j in 2:n) == 1)
end

for k in 2:n
    for i in 2:n
        if i != k
            @constraint(model, sum(z[k, i, j] for j in 1:n if j != i) == sum(z[k, j, i] for j in 1:n if j != i))
        end
    end
end

for k in 2:n
    @constraint(model, sum(z[k, k, j] for j in 1:n if j != k) + 1 == sum(z[k, j, k] for j in 1:n if j != k))
end

for k in 2:n
    for i in 1:n
        for j in 2:n
            if i != j && j != 1
                @constraint(model, z[k, i, j] + z[k, j, i] <= X[i, j] + X[j, i])
            end
        end
    end
end

# Définir les objectifs
@expression(model, obj1, sum(dist_matrix1[i, j] * X[i, j] for i in 1:n for j in 1:n))
@expression(model, obj2, sum(dist_matrix2[i, j] * X[i, j] for i in 1:n for j in 1:n))
@objective(model, Min, [obj1, obj2])

# Optimiser le modèle
optimize!(model)

# Obtenir le nombre de solutions
n_solutions = MOI.get(model, MOI.ResultCount())
println("Solve complete. Found $n_solutions solution(s)")

# Extraire et afficher toutes les solutions
objective_values = []
for i in 1:n_solutions
    push!(objective_values, MOI.get(model, MOI.ObjectiveValue(i)))
end



# Créer un graphique avec toutes les solutions
scatter([x[1] for x in objective_values], [x[2] for x in objective_values])
title("Points supportés")
xlabel("Objectif 1")
ylabel("Objectif 2")

# Afficher le graphique
show()
