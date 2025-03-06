# This is mostly an internal implementation detail that users shouldn't need to worry about.
# We can remove this once https://github.com/FluxML/Optimisers.jl/issues/205 is resolved.
module ReactantCompatibleOptimisers

using ConcreteStructs: @concrete
using Optimisers: Optimisers, AbstractRule

using ..Lux: Lux, Utils

abstract type ReactantCompatibleOptimisersRule <: AbstractRule end

function make_reactant_compatible(opt::AbstractRule)
    return Utils.to_rarray(opt; track_numbers=AbstractFloat)
end

function make_reactant_compatible(opt::Optimisers.RMSProp)
    return Optimisers.RMSProp(
        Utils.to_rarray(opt.eta; track_numbers=AbstractFloat),
        Utils.to_rarray(opt.rho; track_numbers=AbstractFloat),
        opt.epsilon,
        opt.centred
    )
end

function make_reactant_compatible(opt::Optimisers.AdamW)
    return Optimisers.AdamW(
        Utils.to_rarray(opt.eta; track_numbers=AbstractFloat),
        Utils.to_rarray(opt.beta; track_numbers=AbstractFloat),
        Utils.to_rarray(opt.lambda; track_numbers=AbstractFloat),
        opt.epsilon,
        opt.couple
    )
end

function make_reactant_compatible(opt::Optimisers.ClipNorm)
    return Optimisers.ClipNorm(
        Utils.to_rarray(opt.omega; track_numbers=AbstractFloat),
        Utils.to_rarray(opt.p; track_numbers=AbstractFloat),
        opt.throw
    )
end

function make_reactant_compatible(opt::Optimisers.OptimiserChain)
    return Optimisers.OptimiserChain(make_reactant_compatible.(opt.opts))
end

function make_reactant_compatible(opt::Optimisers.AccumGrad)
    return AccumGrad(Utils.to_rarray(opt.n; track_numbers=Integer))
end

@concrete struct AccumGrad <: AbstractRule
    n
end

# XXX: the counter needs to match the client / device?
Optimisers.init(::AccumGrad, x) = zero(x), Utils.to_rarray(1; track_numbers=Integer)

end
