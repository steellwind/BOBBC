using DataStructures # for queue
using LightGraphs
using LightGraphsFlows
using SparseArrays
#for min cut
using PyPlot
using Statistics: mean



@enum PrunedType NONE INFEASIBILITY INTEGRITY DOMINANCE


# ----------------------------------
# ---- SupportedSolutionPoint ------
# ----------------------------------
mutable struct SupportedSolutionPoint
    x::Vector{Vector{Float64}}
    y::Vector{Float64}
    λ :: Vector{Float64}
    is_integer :: Bool 
end

# todo : tolerance
function dominates(a::SupportedSolutionPoint, b::SupportedSolutionPoint)
    if a.y == b.y
        return false
    else 
        return all(a.y .<= b.y)
    end
end

function Base.:show(io::IO, sol::SupportedSolutionPoint)
    println(io, "(y=", sol.y,  
    "; λ=", sol.λ,
    "; is_integer=", sol.is_integer,
    "; x = ", sol.x,
    " )"
    )
end

function Base.:show(io::IO, sols::Vector{SupportedSolutionPoint})
    print(io, "[")
    for sol in sols
        println(io, sol, " , ")
    end
    println(io, "] size=", length(sols))
end

"""
    Return `true` if the given solution point is near to integer under a tolerance.
"""
function _is_integer(algorithm, x::Vector{Float64})::Bool
    tol = MOI.get(algorithm, Tolerance())

    for val in x
        if !(abs(val - floor(Int64, val)) < tol || abs(ceil(Int64, val) - val ) < tol)
            return false
        end
    end
    return true
end

# ----------------------------------
# ---------- Node ------------------
# ----------------------------------
mutable struct Node 
    num::Int64                    
    depth::Int64                # depth in tree
    pred::Node                  # predecessor
    succs::Vector{Node}         # successors
    var_idx::Union{Nothing,MOI.VariableIndex}   # index of the chosen variable to be split
    var_bound::Float64            # variable bound
    bound_type :: Int64         # 1 : <= ; 2 : >= 
    bound_ctr :: Union{Nothing,MOI.ConstraintIndex}    # the branching constraint
    activated::Bool             # if the node is active
    pruned::Bool                # if the node is pruned
    pruned_type::PrunedType      # if the node is fathomed, restore pruned type
    deleted::Bool               # if the node is supposed to be deleted
    lower_bound_set::Vector{SupportedSolutionPoint}        # local lower bound set    
    assignment::Dict{MOI.VariableIndex, Float64}  # (varidex, varbound, boundtype)
    root::Bool

    Node() = new()

    function Node(num::Int64, depth::Int64 ;
        pred::Node=Node(), succs::Vector{Node}=Vector{Node}(), var_idx=nothing, var_bound::Float64=0.0, bound_type::Int64=0, bound_ctr=nothing
   )
        n = new()
        n.num = num
        n.depth = depth
        n.pred = pred
        n.succs = succs
        n.var_idx = var_idx
        n.var_bound = var_bound
        n.bound_type = bound_type
        n.bound_ctr = bound_ctr

        n.activated = true 
        n.pruned = false
        n.pruned_type = NONE
        n.deleted = false
        n.lower_bound_set = Vector{SupportedSolutionPoint}()
        n.assignment = Dict{MOI.VariableIndex, Float64}()

        f(t) = nothing 
        finalizer(f, n)
    end
end

function Base.:show(io::IO, n::Node)
    println(io, "\n\n # ----------- node $(n.num) : \n", 
    "depth = $(n.depth) \n",
    "pred = $(n.pred.num) \n",
    "var[ $(n.var_idx) ] = $(n.var_bound) \n",
    "activated = $(n.activated) \n",
    "pruned = $(n.pruned) \n",
    "pruned_type = $(n.pruned_type)"
    )
    print(io, "succs = [ ")
    for s in n.succs print(io, "$(s.num), ") end
    println(io, " ]")

    println(io, "LBS = ", n.lower_bound_set)
end


"""
Return `true` if the given node is the root of a branch-and-bound tree.
"""
function isRoot(node::Node)
    return node.depth == 0 # !isdefined(node, :pred) 
end

"""
Return `true` if the given `node` has activated/non-explored successor(s).
"""
function hasNonExploredChild(node::Node)
    for c in node.succs
        if c.activated return true end 
    end
    return false
end

"""
Delete the given node in B&B tree. (should be a private function)
"""
function Base.delete!(node::Node)           # todo : check
    node.deleted = true ; node = nothing               # remove from the memory
end

"""
Prune the given node in a B&B tree and delete all successors of the pruned node.
"""
function prune!(node::Node, reason::PrunedType)     # todo : check 
    node.pruned = true
    node.pruned_type = reason

    to_delete = node.succs[:]
    node.succs = Vector{Node}()

    while length(to_delete) > 0
        n = pop!(to_delete)
        to_delete = vcat(to_delete, n.succs[:])
        delete!(n)
    end
end


"""
From the actual node, go up to the root to get the partial assignment of variables.
"""
function getPartialAssign(actual::Node)::Dict{MOI.VariableIndex, Float64} 
    assignment = Dict{MOI.VariableIndex, Float64}()
    if isRoot(actual) # the actual node is the root 
        return assignment
    end
    predecessor = actual.pred
    assignment[actual.var_idx] = actual.var_bound

    while !isRoot(predecessor)     
        actual = predecessor ; predecessor = actual.pred
        if actual.bound_ctr !== nothing
            assignment[actual.var_idx] = actual.var_bound
        end
    end
    return assignment
end


"""
Going through all the predecessors until the root, add variables or objective bounds branched in the predecessors.

Return a list of objective bounds (symbolic constraint).
"""
function setVarBounds(actual::Node, model, Bounds::Vector{Dict{MOI.VariableIndex, MOI.ConstraintIndex}})
    if isRoot(actual) # the actual node is the root 
        return 
    end
    predecessor = actual.pred

    # set actual objective/variable bound
    addOneBound(actual, model, Bounds)

    # set actual objective/variable bounds in predecessors
    while !isRoot(predecessor)    
        actual = predecessor ; predecessor = actual.pred 
        addOneBound(actual, model, Bounds)
    end
end


"""
Remove variables or objective bounds set in the predecessors.
"""
function removeVarBounds(actual::Node, model, Bounds::Vector{Dict{MOI.VariableIndex, MOI.ConstraintIndex}})
    if isRoot(actual) # the actual node is the root 
        return 
    end
    predecessor = actual.pred
    removeOneBound(actual, model, Bounds)

    while !isRoot(predecessor)     
        actual = predecessor ; predecessor = actual.pred
        removeOneBound(actual, model, Bounds)
    end
end

function removeOneBound(actual::Node, model, Bounds::Vector{Dict{MOI.VariableIndex, MOI.ConstraintIndex}})
    MOI.delete(model, actual.bound_ctr)

    if actual.bound_type == 1 
        Bounds[2][actual.var_idx] = MOI.add_constraint(model, actual.var_idx, MOI.LessThan(1.0))
    elseif actual.bound_type == 2
        Bounds[1][actual.var_idx] = MOI.add_constraint(model, actual.var_idx, MOI.GreaterThan(0.0))
    else
        error("bound_type unknown in removeVarBounds() .")
    end
end


"""
Given a partial assignment on variables values, add the corresponding bounds.
"""
function addOneBound(actual::Node, model, Bounds::Vector{Dict{MOI.VariableIndex, MOI.ConstraintIndex}})
    lower_bounds, upper_bounds = Bounds # >= and <= 
    if actual.bound_type == 1 
        MOI.delete(model, upper_bounds[actual.var_idx])
        actual.bound_ctr = MOI.add_constraint(model, actual.var_idx, MOI.LessThan(actual.var_bound))
    elseif actual.bound_type == 2
        MOI.delete(model, lower_bounds[actual.var_idx])
        actual.bound_ctr = MOI.add_constraint(model, actual.var_idx, MOI.GreaterThan(actual.var_bound))
    else
        error("bound_type unknown in addOneBound() .")
    end
end



# ----------------------------------
# ---------- branching -------------
# ----------------------------------
"""
Return an initialized todo list according to the fixed parameter.
"""
function initTree(algorithm)
    if MOI.get(algorithm, TraverseOrder()) == :bfs
        return Queue{Base.RefValue{Node}}()
    elseif MOI.get(algorithm, TraverseOrder()) == :dfs
        return Stack{Base.RefValue{Node}}() 
    elseif MOI.get(algorithm, TraverseOrder()) == :arbitrary
        return Vector{Base.RefValue{Node}}()
    else
        @error "Unknown traverse parameter $(MOI.get(algorithm, TraverseOrder()))\n please set attribute with :bfs, :dfs or :arbitrary ."
    end
end


"""
Add a node identify in the todo list.
"""
function addTree(todo, algorithm, node::Node)
    if MOI.get(algorithm, TraverseOrder()) == :bfs
        enqueue!(todo, Ref(node))
    elseif MOI.get(algorithm, TraverseOrder()) == :dfs
        push!(todo, Ref(node)) 
    elseif MOI.get(algorithm, TraverseOrder()) == :arbitrary
        push!(todo, Ref(node))
    else
        @error "Unknown traverse parameter $(MOI.get(algorithm, TraverseOrder()))\n please set attribute with :bfs, :dfs or :arbitrary ."
    end
end

"""
Return the next element in the todo list.
"""
function nextNodeTree(todo, algorithm)
    if MOI.get(algorithm, TraverseOrder()) == :bfs
        return dequeue!(todo)
    elseif MOI.get(algorithm, TraverseOrder()) == :dfs
        return pop!(todo) 
    elseif MOI.get(algorithm, TraverseOrder()) == :arbitrary
        i = rand(1:length(todo))
        next = todo[i]
        deleteat!(todo, i)
        return next
    else
        @error "Unknown traverse parameter $(MOI.get(algorithm, TraverseOrder()))\n please set attribute with :bfs, :dfs or :arbitrary ."
    end
end

"""
Pick up a free variable to be split according to the prefiexd strategy.
"""
# todo : add other strategies ...
# Fonction pour vérifier si une variable est une variable x
function is_x_variable(var, model::Optimizer)
    var_name = MOI.get(model, MOI.VariableName(), var)
    return startswith(var_name, "X")  # Supposons que les variables X soient nommées avec "X" comme préfixe
end

# Modifier la fonction pickUpAFreeVar pour choisir uniquement les variables x
function pickUpAFreeVar(assignment::Dict{MOI.VariableIndex, Float64}, model) :: Union{Nothing, MOI.VariableIndex}
    # Obtenir les indices des variables
    vars_idx = MOI.get(model, MOI.ListOfVariableIndices())

    if length(assignment) == length(vars_idx)
        return nothing
    end

    # Filtrer uniquement les variables X
    x_vars = [v for v in vars_idx if is_x_variable(v, model)]
    
    # Choisir une variable x qui n'est pas encore assignée
    for var in x_vars
        if !haskey(assignment, var)
            return var
        end
    end

    return nothing
end



# ----------------------------------
# ---------- Bound Sets ------------
# ----------------------------------
function _fix_λ(algorithm, model::Optimizer)
    p = MOI.output_dimension(model.f) ; Λ = []
    for i in 1:p
        λ = zeros(p) ; λ[i] = 1.0
        push!(Λ, λ) 
    end
    λ = [1/p for _ in 1:p] ; push!(Λ, λ)
    
    λ_count = MOI.get(algorithm, LowerBoundsLimit()) - length(Λ)

    # in case of LowerBoundsLimit() > p+1
    for i in 1:p
        if λ_count <= 0 return Λ end
        λ_ = (λ .+ Λ[i]) ./2
        push!(Λ, λ_) ; λ_count -= 1
    end

    while λ_count > 0
        i = rand(1:length(Λ)) ; j = rand(1:length(Λ))
        if i != j
            λ_ = (Λ[j] + Λ[i]) ./2
            push!(Λ, λ_) ; λ_count -= 1
        end
    end

    return Λ
end

# todo : add equivalent solutions
function push_avoiding_duplicate(vec::Vector{SupportedSolutionPoint}, candidate::SupportedSolutionPoint) :: Bool
    for sol in vec
        if sol.y ≈ candidate.y return false end 
    end
    push!(vec, candidate) ; return true
end


## fonction min cut : 

function run_mincut(directed_graph::SimpleDiGraph, sol::SupportedSolutionPoint, NC::Int, capacities::Array{Float64, 2})
    flows = []
    parts = []
    n = Int(sqrt(length(sol.x[1])))  # Nombre de lignes/colonnes dans la matrice

    adj_matrix = reshape(sol.x[1], n, n)  # La matrice d'adjacence est maintenant directement sol.x[1]

    for i in 1:n
        for j in i:n  # Commencer à partir de i pour parcourir seulement la partie diagonale supérieure
            if adj_matrix[i, j] != 0
                if !has_edge(directed_graph, i, j)
                    # Ajouter l'arête dans le sens i -> j
                    add_edge!(directed_graph, i, j)
                    add_edge!(directed_graph, j, i)
                end
            elseif has_edge(directed_graph, i, j)
                # Si l'arête existe mais que sa valeur dans la solution courante est 0, supprimez-la
                rem_edge!(directed_graph, i, j)
                rem_edge!(directed_graph, j, i)
            end
            capacities[i, j] = adj_matrix[i, j]
            capacities[j, i] = adj_matrix[i, j] 
        end
    end
    

    # Vérifier si le graphe est connecté
    if !is_connected(directed_graph)
        # Le graphe n'est pas connecté
        components = connected_components(directed_graph)
        for i in 1:length(components)
            # Prendre un ensemble de sommets non connectés
            part1 = components[i]
            # Prendre tous les autres sommets qui ne font pas partie du premier ensemble comme le deuxième ensemble
            part2 = setdiff(1:n, part1)
            # Vérifiez si le couple (part1, part2) est déjà dans parts
            if !any((p1 == part1 && p2 == part2) || (p1 == part2 && p2 == part1) for (p1, p2) in parts)
                push!(flows, 0)
                push!(parts, (part1, part2))
                if length(flows) >= NC
                    return flows, parts
                end
            end
        end
    end
    

    # Si le graphe est connecté et nous n'avons pas encore atteint NC, compléter avec min-cut
    if length(flows) < NC && is_connected(directed_graph)
        for source in 1:n
            for target in 1:n
                if source != target
                    p1, p2, f = mincut(directed_graph, sol, source, target, capacities)
                    if f < 2
                        # Vérifie si le couple (p1, p2) est déjà dans parts
                        if !any([sort(p1) == sort(x) && sort(p2) == sort(y) for (x, y) in parts])
                            push!(flows, f)
                            push!(parts, (sort(p1), sort(p2)))
                            if length(flows) >= NC
                                return flows, parts
                            end
                        end
                    end
                end
            end
        end
    end

    return flows, parts
end



function mincut(directed_graph::SimpleDiGraph, sol::SupportedSolutionPoint, source::Int, target::Int, capacities::Array{Float64, 2})
    # Obtenir la matrice d'adjacence à partir de sol.x[1]
    n = Int(sqrt(length(sol.x[1])))  # Nombre de lignes/colonnes dans la matrice

    # Appeler maximum_flow avec les nouvelles capacités
    flow, flow_matrix = maximum_flow(directed_graph, source, target, capacities, EdmondsKarpAlgorithm())

    # Calculer la matrice résiduelle
    residual_matrix = capacities - flow_matrix

    # Affichage des arêtes avec leurs capacités
    for edge in edges(directed_graph)
        u, v = src(edge), dst(edge)
        capacity = capacities[u, v]
    end

    # Trouver les deux ensembles de coupure min
    part1 = typeof(source)[]
    queue = [source]
    while !isempty(queue)
        node = pop!(queue)
        push!(part1, node)
        dests = [dst for dst in 1:n if residual_matrix[node, dst] > 0.0 && dst ∉ part1 && dst ∉ queue]
        append!(queue, dests)
    end

    part2 = [node for node in 1:n if node ∉ part1]
    
    return (part1, part2, flow)
end




function add_cuts(model, sol, parts, NC)
    # Supposons que votre matrice d'adjacence est une matrice carrée
    n = Int(sqrt(length(sol.x[1])))
    global total_cut
    for i in 1:NC
        part1, part2 = parts[i]
        terms = MOI.ScalarAffineTerm{Float64}[]
        for v1 in part1
            for v2 in part2
                # Ajustez la façon dont vous calculez l'index pour correspondre à la façon dont les variables ont été ajoutées à votre modèle
                index1 = (v1-1) * n + v2
                index2 = (v2-1) * n + v1
                # Ajoutez un terme pour chaque arête dans les deux directions
                push!(terms, MOI.ScalarAffineTerm(1.0, MOI.VariableIndex(index1)))
                push!(terms, MOI.ScalarAffineTerm(1.0, MOI.VariableIndex(index2)))
            end
        end
        # Créer une fonction affine pour représenter la somme des arêtes
        f = MOI.ScalarAffineFunction(terms, 0.0)
        # Ajouter une contrainte qui oblige la présence de 2 arêtes entre chaque paire d'ensemble de sommets
        total_cut += 1
        MOI.add_constraint(model, f, MOI.GreaterThan(2.0))
    end
end


















"""
Stop looking for lower bounds if duplicate is encounterd
"""


function MOLP(directed_graph::SimpleDiGraph, algorithm, model::Optimizer, node::Node, NC::Int64, capacities::Array{Float64, 2})
    Λ = _fix_λ(algorithm, model)

    constraints = Vector{MOI.ConstraintIndex}()

    for λ in Λ
        status, solution = _solve_weighted_sum(model, Dichotomy(), λ)

        if _is_scalar_status_optimal(status)
            # Convertir solution.x en une matrice d'adjacence
            n = round(Int64, sqrt(length(solution.x)))  # Nombre de sommets
            
            # Extraire et trier les paires clé-valeur en fonction des indices des variables
            sorted_pairs = sort(collect(solution.x), by = x -> x[1].value)
            
            # Collecter les valeurs dans l'ordre trié
            x_vector = collect(x[2] for x in sorted_pairs)
            
            sol = SupportedSolutionPoint([x_vector], solution.y, λ, _is_integer(algorithm, x_vector))
            
            # Exécuter l'algorithme de min-cut
            if NC != 0
                flows, parts = run_mincut(directed_graph, sol, NC, capacities)
                # Ajouter des coupes si flows n'est pas vide
                if !isempty(flows)
                    add_cuts(model, sol, parts, length(flows))
                end
            end

            if any(test -> test.y ≈ sol.y, node.lower_bound_set)
                nothing
            else
                is_new_point = push_avoiding_duplicate(node.lower_bound_set, sol)
                if !is_new_point return end
            end
        end
    end

    if NC!= 0
        node.lower_bound_set = filter_solutions(node.lower_bound_set,model)
    end

    # Visualiser les solutions et les coupes après avoir traité tous les lambdas
    #draw_solution(node.lower_bound_set)
end





function filter_solutions(lower_bound_set::Vector{SupportedSolutionPoint}, model::Optimizer)
    # Récupérez les contraintes du modèle
    constraints = MOI.get(model, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}())
    # Créez un nouveau LBS qui contiendra uniquement les solutions valides
    valid_lower_bound_set = SupportedSolutionPoint[]

    # Parcourez chaque solution dans le LBS
    for sol in lower_bound_set
        is_valid = true

        # Vérifiez chaque contrainte
        for constraint in constraints
            func = MOI.get(model, MOI.ConstraintFunction(), constraint)
            set = MOI.get(model, MOI.ConstraintSet(), constraint)
            # Calculez la valeur de la fonction contrainte pour la solution actuelle
            value = sum(term.coefficient * sol.x[1][term.variable.value] for term in func.terms)

            # Si la valeur ne respecte pas la contrainte, marquez la solution comme invalide et arrêtez de vérifier les autres contraintes
            if value < set.lower
                is_valid = false
                break
            end
        end

        # Si la solution est valide, ajoutez-la au nouveau LBS
        if is_valid
            push!(valid_lower_bound_set, sol)
        end
    end

    # Retournez le nouveau LBS qui contient uniquement les solutions valides
    return valid_lower_bound_set
end




function draw_solution(lower_bound_set::Vector{SupportedSolutionPoint})
    # Supposons que chaque SupportedSolutionPoint a deux critères
    criterion1_values = [sol.y[1] for sol in lower_bound_set]
    criterion2_values = [sol.y[2] for sol in lower_bound_set]

    # Créez un nouveau graphique
    figure()
    
    # Ajoutez les solutions au graphique
    scatter(criterion1_values, criterion2_values, label="Solutions")

    # Ajoutez des légendes et des titres
    title("Solutions dans l'espace des critères")
    xlabel("Critère 1")
    ylabel("Critère 2")
    legend()

    # Affichez le graphique
    show()
end



function testmincut(directed_graph::SimpleDiGraph, sol::SupportedSolutionPoint, capacities::Array{Float64, 2})
    n = Int(sqrt(length(sol.x[1])))  # Nombre de lignes/colonnes dans la matrice

    adj_matrix = reshape(sol.x[1], n, n)  # La matrice d'adjacence est maintenant directement sol.x[1]

    # Mettre à jour les arêtes du SimpleDiGraph à partir de la matrice d'adjacence
    for i in 1:n
        for j in i:n  # Commencer à partir de i pour parcourir seulement la partie diagonale supérieure
            if adj_matrix[i, j] != 0
                if !has_edge(directed_graph, i, j)
                    # Ajouter l'arête dans le sens i -> j
                    add_edge!(directed_graph, i, j)
                    add_edge!(directed_graph, j, i)
                end
            elseif has_edge(directed_graph, i, j)
                # Si l'arête existe mais que sa valeur dans la solution courante est 0, supprimez-la
                rem_edge!(directed_graph, i, j)
                rem_edge!(directed_graph, j, i)
            end
            capacities[i, j] = adj_matrix[i, j]
            capacities[j, i] = adj_matrix[i, j] 
        end
    end
    

    # Vérifier si le mincut est supérieur à 2 pour chaque paire de sommets source et cible
    for source in 1:n
        for target in 1:n
            if source != target
                _, _, f = mincut(directed_graph, sol, source, target, capacities)
                println(f)
                if f < 2
                    # Si le mincut est inférieur ou égal à 2 pour n'importe quelle paire de sommets source et cible, retourner false
                    return false
                end
            end
        end
    end

    # Si le mincut est supérieur à 2 pour toutes les paires de sommets source et cible, retourner true
    return true
end







"""
Compute and stock the relaxed bound set (i.e. the LP relaxation) of the (sub)problem defined by the given node.
Return `true` if the node is pruned by infeasibility.
"""
function computeLBS(directed_graph::SimpleDiGraph, node::Node, model::Optimizer, algorithm, Bounds::Vector{Dict{MOI.VariableIndex, MOI.ConstraintIndex}}, NC::Int64, capacities::Array{Float64, 2})::Bool
    setVarBounds(node, model, Bounds)
    
    MOLP(directed_graph, algorithm, model, node, NC, capacities)

    removeVarBounds(node, model, Bounds)
    return length(node.lower_bound_set) == 0
end

# todo : improve algo complexity ...
function push_filtering_dominance(vec::Vector{SupportedSolutionPoint}, candidate::SupportedSolutionPoint)
    i = 0 ; to_delete = []

    for sol in vec
        i += 1

        if sol.y ≈ candidate.y
            # Point already added to nondominated solutions. Don't add
            for equiv in candidate.x push!(sol.x, equiv) end 
            return
        elseif dominates(sol, candidate)
            # Point is dominated. Don't add
            return
        elseif dominates(candidate, sol)
            # new dominating point 
            push!(to_delete, i)
        end
    end

    deleteat!(vec, to_delete) ; push!(vec, candidate)
    sort!(vec; by = sol ->  sol.y) ; 
end

"""
At the given node, update (filtered by dominance) the global upper bound set.
Return `true` if the node is pruned by integrity.
"""
function updateUBS(node::Node, UBS::Vector{SupportedSolutionPoint}, directed_graph::SimpleDiGraph, capacities::Array{Float64, 2}, NC::Int64)::Bool
    for i in 1:length(node.lower_bound_set) 
        
        if node.lower_bound_set[i].is_integer 
            s = node.lower_bound_set[i] ; push_filtering_dominance(UBS, s)
        end
    end
    return false
end

# ----------------------------------
# ---------- fathoming -------------
# ----------------------------------
"""
Return local nadir points (so-called corner points) of the given UBS.
"""
# todo : p>3 ?!!!!!!!!!!
function getNadirPoints(UBS::Vector{SupportedSolutionPoint}, model) :: Vector{SupportedSolutionPoint}
    p = MOI.output_dimension(model.f)
    nadir_pts = Vector{SupportedSolutionPoint}()

    if length(UBS) == 1 return UBS end 

    if p == 2
        for i in 1:length(UBS)-1
            push!(nadir_pts, SupportedSolutionPoint(Vector{Vector{Float64}}(), 
                                                    [UBS[i+1].y[1], UBS[i].y[2]], 
                                                    Vector{Float64}(), false
                                                )
            )
        end
    else
        nothing
        # todo p > 3
    end
    return nadir_pts
end

"""
A fully explicit dominance test, and prune the given node if it's fathomed by dominance.
(i.e. ∀ l∈L: ∃ u∈U s.t. λu ≤ λl )
Return `true` if the given node is fathomed by dominance.
"""
function fullyExplicitDominanceTest(node::Node, UBS::Vector{SupportedSolutionPoint}, model)
    # we can't compare the LBS and UBS if the incumbent set is empty
    if length(UBS) == 0 return false end

    p = MOI.output_dimension(model.f) ; nadir_pts = getNadirPoints(UBS, model)

    # ------------------------------------------
    # if the LBS consists of a single point
    # ------------------------------------------
    if length(node.lower_bound_set) == 1
        for u ∈ nadir_pts                   # if there exists an upper bound u s.t. u≦l
            if dominates(u, node.lower_bound_set[1])
                return true
            end
        end
        return false
    end

    UBS_ideal = UBS[1].y[:] ; LBS_ideal = node.lower_bound_set[1].y[:]

    for i in 2:length(UBS)
        for z in 1:p
            if UBS[i].y[z] < UBS_ideal[z] UBS_ideal[z] = UBS[i].y[z] end 
        end
    end
    for i in 2:length(node.lower_bound_set)
        for z in 1:p
            if node.lower_bound_set[i].y[z] < LBS_ideal[z] LBS_ideal[z] = node.lower_bound_set[i].y[z] end 
        end
    end
    UBS_ideal_sp = SupportedSolutionPoint(Vector{Vector{Float64}}(), UBS_ideal, Vector{Float64}(), false)
    LBS_ideal_sp = SupportedSolutionPoint(Vector{Vector{Float64}}(), LBS_ideal, Vector{Float64}(), false)

    # ----------------------------------------------
    # if the LBS consists of hyperplanes
    # ----------------------------------------------

    # if only one feasible point in UBS 
    if length(UBS) == 1 
        return dominates(UBS_ideal_sp, LBS_ideal_sp)
    end

    # test range condition necessary 1 : LBS ⊆ UBS (i.e. UBS includes the LP lexico-optimum)
    if !dominates( UBS_ideal_sp, LBS_ideal_sp)  return false end

    # test condition necessary 2 : UBS dominates LBS 
    fathomed = true

    # iterate of all local nadir points
    for u ∈ nadir_pts
        existence = false

        # case 1 : if u is dominates the ideal point of LBS 
        if dominates(u, LBS_ideal_sp)
            return true
        end

        # case 3 : complete pairwise comparison
        for l in node.lower_bound_set             # ∀ l ∈ LBS 
            if l.λ'*u.y < l.λ'*l.y         # todo : add TOL ? 
                existence = true ; break
            end
        end
        
        if !existence return false end
    end

    return true
end

