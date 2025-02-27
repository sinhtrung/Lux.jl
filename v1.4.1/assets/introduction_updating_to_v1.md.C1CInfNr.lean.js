import{_ as a,c as o,a2 as i,o as t}from"./chunks/framework.DPiAi8YZ.js";const p=JSON.parse('{"title":"Updating to Lux v1","description":"","frontmatter":{},"headers":[],"relativePath":"introduction/updating_to_v1.md","filePath":"introduction/updating_to_v1.md","lastUpdated":null}'),r={name:"introduction/updating_to_v1.md"};function n(d,e,l,s,c,u){return t(),o("div",null,e[0]||(e[0]=[i('<h1 id="updating-to-v1" tabindex="-1">Updating to Lux v1 <a class="header-anchor" href="#updating-to-v1" aria-label="Permalink to &quot;Updating to Lux v1 {#updating-to-v1}&quot;">​</a></h1><p>Lux v1 is a Major Release, mostly to signify the stability of the API. In this page, we list out a concrete set of changes that need to be made to your code to update to Lux v1. We also list out some new exciting features that were added as part of this release.</p><h2 id="LuxLib.jl" tabindex="-1"><code>LuxLib.jl</code> <a class="header-anchor" href="#LuxLib.jl" aria-label="Permalink to &quot;`LuxLib.jl` {#LuxLib.jl}&quot;">​</a></h2><h3 id="Breaking-Changes" tabindex="-1">Breaking Changes <a class="header-anchor" href="#Breaking-Changes" aria-label="Permalink to &quot;Breaking Changes {#Breaking-Changes}&quot;">​</a></h3><ul><li><p>Old deprecated API with keyword arguments has been removed. See the new docs in <a href="/v1.4.1/api/NN_Primitives/LuxLib#LuxLib-API">LuxLib API</a> for more details.</p></li><li><p>Default for <a href="/v1.4.1/api/NN_Primitives/LuxLib#LuxLib.API.layernorm"><code>layernorm</code></a> dims has been changed to exclude the batch dimension.</p></li></ul><h3 id="New-Major-Features" tabindex="-1">New Major Features <a class="header-anchor" href="#New-Major-Features" aria-label="Permalink to &quot;New Major Features {#New-Major-Features}&quot;">​</a></h3><ul><li>Dense layers now support CUDA backend for Enzyme (starting <code>v1.1</code>). Wider support for other operations with Enzyme + CUDA is being actively worked on.</li></ul><h2 id="LuxCore.jl" tabindex="-1"><code>LuxCore.jl</code> <a class="header-anchor" href="#LuxCore.jl" aria-label="Permalink to &quot;`LuxCore.jl` {#LuxCore.jl}&quot;">​</a></h2><h3 id="Breaking-Changes-2" tabindex="-1">Breaking Changes <a class="header-anchor" href="#Breaking-Changes-2" aria-label="Permalink to &quot;Breaking Changes {#Breaking-Changes-2}&quot;">​</a></h3><ul><li><p><code>AbstractExplicitLayer</code> has been renamed to <code>AbstractLuxLayer</code>.</p></li><li><p><code>AbstractExplicitContainerLayer</code> behaviour</p><ul><li><p>This has been renamed to <code>AbstractLuxContainerLayer</code>.</p></li><li><p>Previously, <code>AbstractExplicitContainerLayer{(:a,)}</code> (i.e. singleton containers) would produce default initial parameters and states without wrapping them in a <code>NamedTuple{(:a,)}</code>. This was inconsistent with non-singleton containers, and was a source of confusion. With <code>v</code> we return <code>(; a = &lt;parameters&gt;)</code> and <code>(; a = &lt;states&gt;)</code> by default. See <a href="/v1.4.1/api/Building_Blocks/LuxCore#LuxCore.AbstractLuxWrapperLayer"><code>AbstractLuxWrapperLayer</code></a> for a replacement of this functionality.</p></li></ul></li><li><p><code>inputsize</code> has been removed since it was ambiguous and not used anywhere.</p></li><li><p>Changes to <code>outputsize</code>:</p><ul><li><p>Single argument version has been removed. See <a href="https://github.com/LuxDL/LuxCore.jl/pull/43#issuecomment-2254232817" target="_blank" rel="noreferrer">LuxCore.jl Pull Request 43</a> for more details on the rationale behind this change.</p></li><li><p>Fallback implementation has been moved to <code>Lux.jl</code>. (i.e. users using Lux shouldn&#39;t see a difference, but if <code>Lux.jl</code> isn&#39;t loaded, this function has error.)</p><ul><li>Internally this uses a <code>NilArray</code> that is able to compute sizes without actually running the computation.</li></ul></li></ul></li><li><p><code>Functors</code> and <code>Setfield</code> have been made into optional dependencies. Certain <code>LuxCore</code> functionality that rely on these functions, will throw an error if these packages are not loaded.</p></li></ul><h3 id="New-Major-Features-2" tabindex="-1">New Major Features <a class="header-anchor" href="#New-Major-Features-2" aria-label="Permalink to &quot;New Major Features {#New-Major-Features-2}&quot;">​</a></h3><ul><li>Introduction of <a href="/v1.4.1/api/Building_Blocks/LuxCore#LuxCore.AbstractLuxWrapperLayer"><code>AbstractLuxWrapperLayer</code></a>. This behaves exactly like the old singleton container. For example, the old <code>AbstractExplicitContainerLayer{(:a,)}</code> is equivalent to <code>AbstractLuxWrapperLayer{:a}</code>.</li></ul><h2 id="WeightInitializers.jl" tabindex="-1"><code>WeightInitializers.jl</code> <a class="header-anchor" href="#WeightInitializers.jl" aria-label="Permalink to &quot;`WeightInitializers.jl` {#WeightInitializers.jl}&quot;">​</a></h2><p>This was a major release to signify the stability of the API. There were no breaking changes. We do support a wider range of RNG types, see <a href="/v1.4.1/api/Building_Blocks/WeightInitializers#Supported-RNG-Types-WeightInit">Supported RNG Types</a> for more details.</p><h2 id="MLDataDevices.jl" tabindex="-1"><code>MLDataDevices.jl</code> <a class="header-anchor" href="#MLDataDevices.jl" aria-label="Permalink to &quot;`MLDataDevices.jl` {#MLDataDevices.jl}&quot;">​</a></h2><p>This is the most aggressive change that was made. We renamed the <code>LuxDeviceUtils.jl</code> package to <code>MLDataDevices.jl</code>, to allow for non-Lux packages to use this shared device management abstraction.</p><div class="warning custom-block"><p class="custom-block-title">Deprecation of <code>LuxDeviceUtils.jl</code></p><p>This also marks the deprecation of the <code>LuxDeviceUtils.jl</code> package. We won&#39;t be making any updates to that package, including fixing any bugs. All users should switch to <code>MLDataDevices.jl</code> instead.</p></div><h3 id="Breaking-Changes-3" tabindex="-1">Breaking Changes <a class="header-anchor" href="#Breaking-Changes-3" aria-label="Permalink to &quot;Breaking Changes {#Breaking-Changes-3}&quot;">​</a></h3><ul><li><p><code>Lux(___)Device</code> objects have been renamed to <code>(___)Device</code>. For example, <code>LuxCUDADevice</code> has been renamed to <code>CUDADevice</code>.</p></li><li><p><code>Lux(___)Adaptor</code> objects have been removed. The corresponding <code>Device</code> objects should be used directly instead.</p></li></ul><h3 id="New-Major-Features-3" tabindex="-1">New Major Features <a class="header-anchor" href="#New-Major-Features-3" aria-label="Permalink to &quot;New Major Features {#New-Major-Features-3}&quot;">​</a></h3><ul><li><a href="/v1.4.1/api/Accelerator_Support/MLDataDevices#MLDataDevices.DeviceIterator"><code>DeviceIterator</code></a> provides a generalization of <code>CUDA.CuIterator</code> and works for all backends and more data types (using <code>Functors.jl</code>). <code>MLUtils.DataLoader |&gt; gdev</code> now returns a <code>DeviceIterator</code> instead of being a no-op.</li></ul><h2 id="Lux.jl" tabindex="-1"><code>Lux.jl</code> <a class="header-anchor" href="#Lux.jl" aria-label="Permalink to &quot;`Lux.jl` {#Lux.jl}&quot;">​</a></h2><h3 id="Breaking-Changes-(Removed-Functionality)" tabindex="-1">Breaking Changes (Removed Functionality) <a class="header-anchor" href="#Breaking-Changes-(Removed-Functionality)" aria-label="Permalink to &quot;Breaking Changes (Removed Functionality) {#Breaking-Changes-(Removed-Functionality)}&quot;">​</a></h3><ul><li><p>Direct reexport of <code>NNlib</code> has been removed. We reexport selected functionality from <code>NNlib</code>. Direactly load <code>NNlib</code> if you need to use the other functions.</p></li><li><p>Flattening of <a href="/v1.4.1/api/Lux/layers#Lux.Chain"><code>Chain</code></a> layers has been removed, and the corresponding <code>disable_optimizations</code> kwarg has been removed.</p></li><li><p>Some layers overloaded <code>Base.keys</code>, these have been removed. These were mostly un-documented and weren&#39;t supposed to be used outside of the <code>Lux.jl</code> package.</p></li><li><p><a href="/v1.4.1/api/Lux/utilities#Lux.Training.TrainState"><code>Training.TrainState</code></a> construction with <code>rng</code> has been removed.</p></li><li><p>Older versions of Preferences have been removed.</p></li><li><p><code>disable_stacktrace_truncation!</code> has been removed. From Julia 1.9 onwards, stacktrace truncation is enabled by default.</p></li><li><p>Certain Experimental features were present outside the <code>Lux.Experimental</code> module. These have been removed, use them via <code>Lux.Experimental</code> instead. Run Julia with with <code>depwarn</code> as <code>error</code> and Lux <code>v0.5</code> to see the deprecations.</p></li><li><p><code>Lux.Experimental.@layer_map</code> is not longer needed and has been removed. The name of the variable prevents writing generic functions and is no longer pre-pended to the <code>KeyPath</code>. See the docstring of <a href="/v1.4.1/api/Lux/contrib#Lux.Experimental.layer_map"><code>Lux.Experimental.layer_map</code></a> for more details.</p></li><li><p><code>allow_fast_activation</code> kwarg has been removed completely. Pass an anonymous function as the activation to prevent internal modivations to the activation function.</p></li></ul><h3 id="Breaking-Changes-(Moved-Functionality)" tabindex="-1">Breaking Changes (Moved Functionality) <a class="header-anchor" href="#Breaking-Changes-(Moved-Functionality)" aria-label="Permalink to &quot;Breaking Changes (Moved Functionality) {#Breaking-Changes-(Moved-Functionality)}&quot;">​</a></h3><ul><li><p><code>Lux.Experimental.Training</code> has been moved to <code>Lux.Training</code>. We guarantee SemVar on this new module.</p></li><li><p><code>Lux.cpu</code> and <code>Lux.gpu</code> have been removed. Use <a href="/v1.4.1/api/Accelerator_Support/MLDataDevices#MLDataDevices.cpu_device"><code>cpu_device</code></a> and <a href="/v1.4.1/api/Accelerator_Support/MLDataDevices#MLDataDevices.gpu_device"><code>gpu_device</code></a> instead.</p></li><li><p><code>Experimental.@compact</code> can be directly used via <a href="/v1.4.1/api/Lux/utilities#Lux.@compact"><code>@compact</code></a> now.</p></li><li><p><code>Experimental.StatefulLuxLayer</code> has been moved to <a href="/v1.4.1/api/Lux/utilities#Lux.StatefulLuxLayer"><code>Lux.StatefulLuxLayer</code></a>.</p></li><li><p><code>st_fixed_path</code> kwarg has been removed from <a href="/v1.4.1/api/Lux/utilities#Lux.StatefulLuxLayer"><code>Lux.StatefulLuxLayer</code></a>, instead use it as <code>StatefulLuxLayer{st_fixed_path}(...)</code>.</p></li><li><p>Strings as inputs to <a href="/v1.4.1/api/Lux/contrib#Lux.Experimental.layer_map"><code>Lux.Experimental.layer_map</code></a> and <a href="/v1.4.1/api/Lux/contrib#Lux.Experimental.@debug_mode"><code>Lux.Experimental.@debug_mode</code></a> are removed, use <code>Functors.KeyPath</code> instead.</p></li><li><p><code>CrossCor</code> has been removed. Use <code>Conv(args...; kwargs..., cross_correlation=true)</code> instead.</p></li></ul><h3 id="Breaking-Changes-(Changes-in-Defaults)" tabindex="-1">Breaking Changes (Changes in Defaults) <a class="header-anchor" href="#Breaking-Changes-(Changes-in-Defaults)" aria-label="Permalink to &quot;Breaking Changes (Changes in Defaults) {#Breaking-Changes-(Changes-in-Defaults)}&quot;">​</a></h3><ul><li><p><a href="/v1.4.1/api/Lux/layers#Lux.Conv"><code>Conv</code></a> and <a href="/v1.4.1/api/Lux/layers#Lux.ConvTranspose"><code>ConvTranspose</code></a> use an initialization based on the activation function, taken from Pytorch. Pytorch assumes the activation function is <code>leakyrelu</code> to compute the gain, however, we compute the gain based on the activation function passed in to the layer.</p></li><li><p><a href="/v1.4.1/api/Lux/layers#Lux.Upsample"><code>Upsample</code></a> now has an <code>align_corners</code> keyword argument, which defaults to <code>false</code>. Previously this was always <code>true</code>.</p></li><li><p><a href="/v1.4.1/api/Lux/layers#Lux.Dense"><code>Dense</code></a> and <a href="/v1.4.1/api/Lux/layers#Lux.Bilinear"><code>Bilinear</code></a> have updated default initializations to align with the defaults from Pytorch. See the documentation for more details.</p></li><li><p><a href="/v1.4.1/api/Lux/layers#Lux.InstanceNorm"><code>InstanceNorm</code></a> now defaults to <code>affine=false</code> instead of <code>affine=true</code>.</p></li><li><p><a href="/v1.4.1/api/Lux/layers#Lux.Embedding"><code>Embedding</code></a> now defaults to <code>init_weight=rand32</code> instead of <code>init_weight=randn32</code>.</p></li><li><p>Recurrent Cells - <a href="/v1.4.1/api/Lux/layers#Lux.RNNCell"><code>RNNCell</code></a>, <a href="/v1.4.1/api/Lux/layers#Lux.LSTMCell"><code>LSTMCell</code></a>, and <a href="/v1.4.1/api/Lux/layers#Lux.GRUCell"><code>GRUCell</code></a> now have different default initializations. See the documentation for more details.</p></li></ul><h3 id="New-Features" tabindex="-1">New Features <a class="header-anchor" href="#New-Features" aria-label="Permalink to &quot;New Features {#New-Features}&quot;">​</a></h3><ul><li><p><a href="/v1.4.1/api/Lux/layers#Lux.InstanceNorm"><code>InstanceNorm</code></a> now supports tracking statistics.</p></li><li><p><a href="/v1.4.1/api/Lux/layers#Lux.RNNCell"><code>RNNCell</code></a> and <a href="/v1.4.1/api/Lux/layers#Lux.LSTMCell"><code>LSTMCell</code></a> add <code>bias_ih</code> and <code>bias_hh</code> to the parameters to align with Pytorch. Both are controlled using <code>init_bias</code> and <code>use_bias</code>.</p></li><li><p><a href="/v1.4.1/api/Lux/layers#Lux.ConvTranspose"><code>ConvTranspose</code></a> allows <code>flipkernel=true</code> via <code>cross_correlation=true</code>. This makes it efficient for MIOpen.</p></li><li><p><a href="/v1.4.1/api/Lux/layers#Lux.ConvTranspose"><code>ConvTranspose</code></a> now has an <code>outpad</code> keyword argument, which is used to increase the size of the output in the desired dimensions.</p></li><li><p>Pooling Layers based on lpnorm have been added – <a href="/v1.4.1/api/Lux/layers#Lux.LPPool"><code>LPPool</code></a>, <a href="/v1.4.1/api/Lux/layers#Lux.GlobalLPPool"><code>GlobalLPPool</code></a>, and <a href="/v1.4.1/api/Lux/layers#Lux.AdaptiveLPPool"><code>AdaptiveLPPool</code></a>.</p></li></ul>',30)]))}const L=a(r,[["render",n]]);export{p as __pageData,L as default};
