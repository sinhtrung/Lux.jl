Utils.vec(x::AnyTracedRArray) = Reactant.TracedUtils.materialize_traced_array(vec(x))

# XXX: Use PoolDims once EnzymeJAX supports stablehlo.reduce_window adjoint
Lux.calculate_pool_dims(g::Lux.GlobalPoolMode, ::TracedRArray) = g

# Optimisers AccumGrad
function Optimisers.apply!(opt::Lux.ReactantCompatibleOptimisers.AccumGrad, state, x, dx)
    accum_dx, counter = state
    @. accum_dx += dx / opt.n
    @trace if counter == opt.n
        dx_final = dx
        counter = 1
    else
        dx_final = zero.(dx)
        counter += 1
    end
    return (accum_dx, counter), dx_final
end

function Lux.ReactantCompatibleOptimisers.setup_optimiser_with_jit(opt, ps)
    return @jit Optimisers.setup(opt, ps)
end
