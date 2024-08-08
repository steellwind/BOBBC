using DataStructures # for queue
using LightGraphs
using LightGraphsFlows
using SparseArrays
#for min cut
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

function run_mincut(directed_graph::SimpleDiGraph, sol::SupportedSolutionPoint, NC::Int, capacities::Array{Float64, 2}, n::Int64)
    flows = []
    parts = []

    adj_matrix = transpose(reshape(sol.x[1], n, n))# La matrice d'adjacence est maintenant directement sol.x[1]
    # Rendre la matrice symétrique
    for i in 1:n
        for j in (i+1):n
            adj_matrix[i, j] = adj_matrix[j, i] = adj_matrix[i, j]+ adj_matrix[j, i]
        end
    end

    # Mettre à jour les arêtes du SimpleDiGraph à partir de la matrice d'adjacence
    for i in 1:n
        for j in 1:n  # Parcourir toutes les paires de sommets
            if adj_matrix[i, j] != 0
                if !has_edge(directed_graph, i, j)
                    # Ajouter l'arête dans le sens i -> j
                    add_edge!(directed_graph, i, j)
                end
            elseif has_edge(directed_graph, i, j)
                # Si l'arête existe mais que sa valeur dans la solution courante est 0, supprimez-la
                rem_edge!(directed_graph, i, j)
            end
            capacities[i, j] = adj_matrix[i, j]
        end
    end
    

    if !is_connected(directed_graph)
        # Le graphe n'est pas connecté
        components = connected_components(directed_graph)
        for i in 1:length(components)
            # Prendre un ensemble de sommets non connectés
            part1 = components[i]
            # Tous les autres sommets forment la deuxième partie
            part2 = setdiff(vertices(directed_graph), part1)
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
                    p1, p2, f = mincut(directed_graph, sol, source, target, capacities,n)
                    if f < 2 - 1e-3
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



function mincut(directed_graph::SimpleDiGraph, sol::SupportedSolutionPoint, source::Int, target::Int, capacities::Array{Float64, 2}, n::Int64)


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




function add_cuts(model, parts, NC, n)

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


# fonction pour remplir l'ubs avec l'heuristique :


function fillUBS(UBS::Vector{SupportedSolutionPoint}, lower_bound_set::Vector{SupportedSolutionPoint}, model::Optimizer)
    # Transformer chaque solution dans lower_bound_set en un chemin valide de TSP
    transformed_solutions = [primalHeuristicTSP(s,model) for s in lower_bound_set]

    # Pour chaque solution transformée, l'ajouter à UBS en utilisant push_filtering_dominance
    for s in transformed_solutions
        push_filtering_dominance(UBS, s)

    end

    return UBS
end


#fonction pour ajouter les coupes au model aprés le cacul des ubs

function mangercut(directed_graph::SimpleDiGraph, lower_bound_set::Vector{SupportedSolutionPoint}, model::Optimizer, capacities::Array{Float64, 2}, NC::Int64)
    # Parcourir toutes les solutions du lbs
    n = Int(sqrt(length(MOI.get(model, MOI.ListOfVariableIndices()))))  # Nombre de sommets
    cuts_added = false

    i = 1
    while i <= length(lower_bound_set)
        sol = lower_bound_set[i]

        # Imprimer la solution actuelle du LBS
        println("Solution actuelle du LBS : ", sol.y)

        # Exécuter l'algorithme de min-cut
        if NC != 0
            flows, parts = run_mincut(directed_graph, sol, NC, capacities, n)
            # Ajouter des coupes si flows n'est pas vide
            if !isempty(flows)
                println(flows)   
                println("partieee ", parts)         
                add_cuts(model, parts, length(flows),n)
                cuts_added = true

                # Supprimer la solution du LBS
                println("DELEATTTTTTTTTTTTTTTTTTTTTTTTTT")
                deleteat!(lower_bound_set, i)
            else
                i += 1
            end
        end
    end

    return(cuts_added)
end






"""
Stop looking for lower bounds if duplicate is encounterd
"""


function MOLP(algorithm, model::Optimizer, node::Node)
    Λ = _fix_λ(algorithm, model)

    constraints = Vector{MOI.ConstraintIndex}()


    for λ in Λ
        status, solution = _solve_weighted_sum(model, Dichotomy(), λ)
        if _is_scalar_status_optimal(status)

            # Extraire et trier les paires clé-valeur en fonction des indices des variables
            sorted_pairs = sort(collect(solution.x), by = x -> x[1].value)
            
            # Collecter les valeurs dans l'ordre trié
            x_vector = collect(x[2] for x in sorted_pairs)
            
            sol = SupportedSolutionPoint([x_vector], solution.y, λ, _is_integer(algorithm, x_vector))
            

            if any(test -> test.y ≈ sol.y, node.lower_bound_set)
                nothing
            else
                is_new_point = push_avoiding_duplicate(node.lower_bound_set, sol)
                if !is_new_point return end
            end
        end
    end



end

##test de solutions.

function is_valid_tsp_path(sol::SupportedSolutionPoint)
    n = round(Int64, sqrt(length(sol.x[1])))  # Nombre de sommets
    path_matrix = transpose(reshape(sol.x[1], n, n))  # Matrice d'adjacence

    visited = zeros(Bool, n)  # Tableau pour suivre les sommets visités

    # Fonction DFS pour parcourir le graphe
    function dfs(node)
        visited[node] = true
        for i in 1:n
            if path_matrix[node, i] != 0 && !visited[i]
                dfs(i)
            end
        end
    end

    # Appel de la fonction DFS à partir du premier sommet
    dfs(1)

    # Vérifier si tous les sommets ont été visités
    for i in 1:n
        if !visited[i]
            return false
        end
    end

    return true
end


## heuristique primal
function primalHeuristicTSP(sol::SupportedSolutionPoint, model::Optimizer)
    # Get the x value from sol
    n = round(Int64, sqrt(length(sol.x[1]))) # Nombre de lignes/colonnes dans la matrice

    xfrac = transpose(reshape(sol.x[1], n, n))
    λ = sol.λ
    # Initialize the solution matrix
    sol=zeros(Float64,n, n)
        
    L=[]
    for i in 1:n
        for j in i+1:n
            push!(L,(i,j,xfrac[i,j]))
        end
    end
    sort!(L,by = x -> x[3])  
       
    CC= zeros(Int64,n)  #Connected component of node i
    for i in 1:n
        CC[i]=-1
    end

    tour=zeros(Int64,n,2)  # the two neighbours of i in a TSP tour, the first is always filled before de second
    for i in 1:n
        tour[i,1]=-1
        tour[i,2]=-1
    end
     
    cpt=0
    while ( (cpt!=n-1) && (size(L)!=0) )
        (i,j,val)=pop!(L)   

        if ( ( (CC[i]==-1) || (CC[j]==-1) || (CC[i]!=CC[j]) )  && (tour[i,2]==-1) && (tour[j,2]==-1) ) 
            cpt=cpt+1 
           
            if (tour[i,1]==-1)  # if no edge going out from i in the sol
                tour[i,1]=j        # the first outgoing edge is j
                CC[i]=i
            else
                tour[i,2]=j        # otherwise the second outgoing edge is j
            end

            if (tour[j,1]==-1)
                tour[j,1]=i
                CC[j]=CC[i]
            else
                tour[j,2]=i
        	
                oldi=i
                k=j
                while (tour[k,2]!=-1)  # update to i the CC of all the nodes linked to j
                    if (tour[k,2]==oldi) 
                        l=tour[k,1]
                    else 
                        l=tour[k,2]
                    end
                    CC[l]=CC[i]
                    oldi=k
                    k=l
                end
            end
        end
    end
     
    i1=-1          # two nodes haven't their 2nd neighbour encoded at the end of the previous loop
    i2=0
    for i in 1:n
        if tour[i,2]==-1
            if i1==-1
                i1=i
            else 
                i2=i
            end
        end
    end
    tour[i1,2]=i2
    tour[i2,2]=i1
    
    for i in 1:n
        for j in i+1:n     
            if ((j!=tour[i,1])&&(j!=tour[i,2]))
                sol[i,j]=0
            else          
                sol[i,j]=1      
            end
        end
    end
      
    # Remodeler la matrice de solution en un vecteur
    sol_vector = vec(reshape(sol, 1, n*n))

    # Récupérer les fonctions objectif du modèle
    objective_functions = model.f

    # Initialiser les valeurs des objectifs à 0
    objective_value1 = 0.0
    objective_value2 = 0.0

    # Parcourir chaque terme de la première fonction objectif
    for i in 1:n*(n-1)
        # Extraire l'indice de la variable et son coefficient
        var_index = objective_functions.terms[i].scalar_term.variable.value
        coefficient = objective_functions.terms[i].scalar_term.coefficient

        # Ajouter le produit du coefficient et de la valeur de la variable à la valeur de l'objectif
        objective_value1 += coefficient * sol_vector[var_index]
    end

    # Parcourir chaque terme de la deuxième fonction objectif
    for i in n*(n-1)+1:2*n*(n-1)
        # Extraire l'indice de la variable et son coefficient
        var_index = objective_functions.terms[i].scalar_term.variable.value
        coefficient = objective_functions.terms[i].scalar_term.coefficient

        # Ajouter le produit du coefficient et de la valeur de la variable à la valeur de l'objectif
        objective_value2 += coefficient * sol_vector[var_index]
    end

    # Imprimer le vecteur de solution et les valeurs des objectifs
    # println("Vecteur de solution : ", sol_vector)
    # println("Valeur de l'objectif 1 : ", objective_value1)
    # println("Valeur de l'objectif 2 : ", objective_value2)

    # Créez une nouvelle instance de SupportedSolutionPoint
    new_sol = SupportedSolutionPoint([sol_vector], [objective_value1, objective_value2], λ, true)

    return new_sol


end







"""
Compute and stock the relaxed bound set (i.e. the LP relaxation) of the (sub)problem defined by the given node.
Return `true` if the node is pruned by infeasibility.
"""
function computeLBS(node::Node, model::Optimizer, algorithm, Bounds::Vector{Dict{MOI.VariableIndex, MOI.ConstraintIndex}})::Bool
    setVarBounds(node, model, Bounds)
    
    MOLP(algorithm, model, node)

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
function updateUBS(node::Node, UBS::Vector{SupportedSolutionPoint})::Bool
    for i in 1:length(node.lower_bound_set)        
        if node.lower_bound_set[i].is_integer 
            s = node.lower_bound_set[i]
            # Ajouter une solution à l'UBS seulement si c'est un chemin valide pour le TSP
            if is_valid_tsp_path(s)
                push_filtering_dominance(UBS, s)
            end
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

