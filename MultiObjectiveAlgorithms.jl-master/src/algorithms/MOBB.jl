include("MOBBTree.jl")



"""
   MultiObjectiveBranchBound()

`MultiObjectiveBranchBound` implements the multi-objective branch&bound framework.

## Supported optimizer attributes

* `MOA.LowerBoundsLimit()`: the maximum number of lower bounds calculated at each B&B node.

 ## Hypothesis :

 * only consider BINARY LINEAR programs for now (but not limited to) # todo : change branching strategy 

 * no not deal with objective with type `FEASIBILITY_SENSE`

"""

global total_cut = 0


mutable struct MultiObjectiveBranchBound <: AbstractAlgorithm
    lowerbounds_limit::Union{Nothing,Int}                   # the number of lower bounds solved at each node 
    traverse_order :: Union{Nothing, Symbol}                # the traversing order of B&B tree
    tolerance :: Union{Nothing, Float64}                    # numerical tolerance

    # --------------- informations for getting attributes 
    pruned_nodes :: Union{Nothing, Int64}

    MultiObjectiveBranchBound() = new(nothing, nothing, nothing,
                                      nothing
                                )
end

# -------------------------------------
# ----------- parameters --------------
# -------------------------------------
MOI.supports(::MultiObjectiveBranchBound, ::LowerBoundsLimit) = true

function MOI.set(alg::MultiObjectiveBranchBound, ::LowerBoundsLimit, value)
    alg.lowerbounds_limit = value ; return
end

function MOI.get(alg::MultiObjectiveBranchBound, attr::LowerBoundsLimit)
    return something(alg.lowerbounds_limit, default(alg, attr))
end

MOI.supports(::MultiObjectiveBranchBound, ::TraverseOrder) = true

function MOI.set(alg::MultiObjectiveBranchBound, ::TraverseOrder, order)
    alg.traverse_order = order ; return
end

function MOI.get(alg::MultiObjectiveBranchBound, attr::TraverseOrder)
    return something(alg.traverse_order, default(alg, attr))
end

MOI.supports(::MultiObjectiveBranchBound, ::Tolerance) = true

function MOI.set(alg::MultiObjectiveBranchBound, ::Tolerance, tol)
    alg.tolerance = tol ; return
end

function MOI.get(alg::MultiObjectiveBranchBound, attr::Tolerance)
    return something(alg.tolerance, default(alg, attr))
end

# --------- attributes only for getting 
MOI.supports(::MultiObjectiveBranchBound, ::PrunedNodeCount) = true

function MOI.get(alg::MultiObjectiveBranchBound, attr::PrunedNodeCount)
    return something(alg.pruned_nodes, default(alg, attr))
end

"""
    Relax binary variables to continuous between 0.0 and 1.0.
"""
function relaxVariables(model::Optimizer) :: Vector{Dict{MOI.VariableIndex, MOI.ConstraintIndex}}
    vars_idx = MOI.get(model, MOI.ListOfVariableIndices())

    # todo : to complete constraints type see  https://jump.dev/MathOptInterface.jl/stable/manual/constraints/
    for (t1, t2) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        ctr_t =  MOI.get(model, MOI.ListOfConstraintIndices{t1,t2}())

        if t1 == MOI.VariableIndex
            for ci in ctr_t
                MOI.delete(model, ci)
            end
        end
    end

    lower_bounds = Dict{MOI.VariableIndex, MOI.ConstraintIndex}(
        var => MOI.add_constraint(model, var, MOI.GreaterThan(0.0)) for var in vars_idx
    )
    
    upper_bounds = Dict{MOI.VariableIndex, MOI.ConstraintIndex}(
        var => MOI.add_constraint(model, var, MOI.LessThan(1.0)) for var in vars_idx
    )

    return [lower_bounds, upper_bounds]
end


## ajout de mes fonction : 

function convert_to_supported(algorithm, solutions::Vector{Main.MultiObjectiveAlgorithms.SolutionPoint}, num_sommets::Int)
    supported_solutions = Vector{SupportedSolutionPoint}()
    
    for i in 1:length(solutions)
        
        solution = solutions[i]


        # Extraire et trier les paires clé-valeur en fonction des indices des variables
        sorted_pairs = sort(collect(solution.x), by = x -> x[1].value)
        
        # Collecter les valeurs dans l'ordre trié
        x_vector = collect(x[2] for x in sorted_pairs)
        # Calculer le vecteur λ
        if i == 1
            λ = [1.0, 0.0]  # première extrémité
        elseif i == length(solutions)
            λ = [0.0, 1.0]  # deuxième extrémité
        else
            next_solution = solutions[i+1]
            λ = next_solution.y - solution.y
        end
        
        # Vérifier si la solution est entière
        is_integer = _is_integer(algorithm, x_vector)
          

        push!(supported_solutions, SupportedSolutionPoint(
            [x_vector],  
            solution.y,
            λ,
            is_integer
        ))
    end
    
    return supported_solutions
end







##



function MOBB(
    directed_graph::SimpleDiGraph,
    algorithm::MultiObjectiveBranchBound,
    model::Optimizer,
    Bounds::Vector{Dict{MOI.VariableIndex, MOI.ConstraintIndex}},
    tree,
    node::Node,
    UBS::Vector{SupportedSolutionPoint},
    NC::Int64,
    capacities::Array{Float64, 2}
)
    # Vérifier que le nœud est activé
    @assert node.activated == true "Le nœud actuel n'est pas activé"
    node.activated = false

    # Print des solutions dans UBS
    println("Node $(node.num): Current solutions in UBS")
    for (i, sol) in enumerate(UBS)
        println("Objectives: $(sol.y)")
    end


    # Calculer l'ensemble des bornes inférieures locales
    if computeLBS(directed_graph, node, model, algorithm, Bounds, NC, capacities)
        prune!(node, INFEASIBILITY)
        algorithm.pruned_nodes += 1
        return
    end

    # Mettre à jour l'ensemble des bornes supérieures
    if updateUBS(node, UBS, directed_graph, capacities, NC)
        algorithm.pruned_nodes += 1
        return
    end

    # Test de dominance complet
    if fullyExplicitDominanceTest(node, UBS, model)
        prune!(node, DOMINANCE)
        algorithm.pruned_nodes += 1
        return
    end

    # Sinon, ce nœud n'est pas élagué, continuer à se brancher sur la variable libre
    assignment = getPartialAssign(node)
    var = pickUpAFreeVar(assignment, model)
    if var === nothing
        return
    end

    children = [
        Node(model.total_nodes + 1, node.depth + 1, pred=node, var_idx=var, var_bound=1.0, bound_type=2),
        Node(model.total_nodes + 2, node.depth + 1, pred=node, var_idx=var, var_bound=0.0, bound_type=1)
    ]
    for child in children
        addTree(tree, algorithm, child)
        model.total_nodes += 1
        push!(node.succs, child)
    end
end





# -------------------------------------
# ----------- main program ------------
# -------------------------------------

function optimize_multiobjective!(
    algorithm::MultiObjectiveBranchBound,
    model::Optimizer,
    # verbose :: Bool = false,
)   
    NC = 2
    global total_cut
    model.total_nodes = 0
    algorithm.pruned_nodes = 0
    start_time = time()
    n_sommet = round(Int64,sqrt(MOI.get(model, MOI.NumberOfVariables())))
    directed_graph = SimpleDiGraph(n_sommet) #initialiser le graphe
    capacities = zeros(Float64, n_sommet, n_sommet) #les capacités


    # step1 - set tolerance to inner model 
    if MOI.get(algorithm, Tolerance()) != default(algorithm, Tolerance())
        MOI.set(model, MOI.RawOptimizerAttribute("tol_inconsistent"), MOI.get(algorithm, Tolerance()))
    end
    
    # step2 - check lower bounds limit 
    if MOI.get(algorithm, LowerBoundsLimit()) < MOI.output_dimension(model.f)
        # at least p lower bounds optimized on each objective 
        MOI.set(algorithm, LowerBoundsLimit(), MOI.output_dimension(model.f) + 1)
    end

    # step3 - LP relaxation 
    Bounds = relaxVariables(model)

    # step4 - initialization
    UBS = Vector{SupportedSolutionPoint}()
    tree = initTree(algorithm)
    model.total_nodes += 1
    root = Node(model.total_nodes, 0)
    addTree(tree, algorithm, root)

    # Utiliser l'algorithme dichotomique pour le premier nœud
    dichotomy_algorithm = Dichotomy()
    status, initial_solutions = optimize_multiobjective!(dichotomy_algorithm, model)

    if status != MOI.OPTIMAL
        return status, []
    end

    # Convertir les solutions initiales en SupportedSolutionPoint et les ajouter dans le lower_bound_set du noeud
    initial_supported_solutions = convert_to_supported(algorithm, initial_solutions, n_sommet)

    # Compléter le LBS
    for sol in initial_supported_solutions
        #push!(root.lower_bound_set, sol)
    # Exécuter l'algorithme de min-cut pour chaque solution initiale
        #flows, parts = run_mincut(directed_graph, sol, NC, capacities)
    # Ajouter des coupes si flows n'est pas vide
        #if !isempty(flows)
            #add_cuts(model, sol, parts, length(flows))
        #end
    end

    status = MOI.OPTIMAL

    # step5 - study every node in tree 
    while length(tree) > 0
        if _time_limit_exceeded(model, start_time)
            status = MOI.TIME_LIMIT
            break
        end

        node_ref = nextNodeTree(tree, algorithm)

        MOBB(directed_graph ,algorithm, model, Bounds, tree, node_ref[], UBS, NC, capacities)
        
        if node_ref[].deleted
            finalize(node_ref[])
        end
    end
    
    vars_idx = MOI.get(model, MOI.ListOfVariableIndices())
    println("nombre de cut total ", total_cut)
    return status, [SolutionPoint(
                    Dict(vars_idx[i] => sol.x[1][i] for i in 1:length(vars_idx)), sol.y
                   ) for sol in UBS]
end


