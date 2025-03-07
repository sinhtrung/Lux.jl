import{_ as s,c as n,o as e,al as p}from"./chunks/framework.BCN3FD2k.js";const d=JSON.parse('{"title":"Graph Convolutional Networks on Cora","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/intermediate/6_GCN_Cora.md","filePath":"tutorials/intermediate/6_GCN_Cora.md","lastUpdated":null}'),c={name:"tutorials/intermediate/6_GCN_Cora.md"};function i(t,a,r,l,f,o){return e(),n("div",null,a[0]||(a[0]=[p(`<h1 id="GCN-Tutorial-Cora" tabindex="-1">Graph Convolutional Networks on Cora <a class="header-anchor" href="#GCN-Tutorial-Cora" aria-label="Permalink to &quot;Graph Convolutional Networks on Cora {#GCN-Tutorial-Cora}&quot;">​</a></h1><p>This example is based on <a href="https://github.com/ml-explore/mlx-examples/blob/main/gcn/" target="_blank" rel="noreferrer">GCN MLX tutorial</a>. While we are doing this manually, we recommend directly using <a href="https://juliagraphs.org/GraphNeuralNetworks.jl/docs/GNNLux.jl/stable/" target="_blank" rel="noreferrer">GNNLux.jl</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux, Reactant, MLDatasets, Random, Statistics, Enzyme, GNNGraphs, ConcreteStructs,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Printf, OneHotArrays, Optimisers</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> reactant_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; force </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cdev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(::MLDataDevices.CPUDevice) (generic function with 1 method)</span></span></code></pre></div><h2 id="Loading-Cora-Dataset" tabindex="-1">Loading Cora Dataset <a class="header-anchor" href="#Loading-Cora-Dataset" aria-label="Permalink to &quot;Loading Cora Dataset {#Loading-Cora-Dataset}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Cora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">graphs[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gnngraph </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GNNGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_index; ndata </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data, edata </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">edge_data, gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">num_nodes</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">features,</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        onehotbatch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">node_data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">targets, data</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">metadata[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;classes&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use a dense matrix here to avoid incompatibility with Reactant</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">adjacency_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gnngraph)),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # We use this since Reactant doesn&#39;t yet support gather adjoint</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">140</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">141</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">640</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1709</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2708</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loadcora (generic function with 1 method)</span></span></code></pre></div><h2 id="Model-Definition" tabindex="-1">Model Definition <a class="header-anchor" href="#Model-Definition" aria-label="Permalink to &quot;Model Definition {#Model-Definition}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dense </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(args</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> adj</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, h_dim, out_dim; nb_layers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dropout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    layer_sizes </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_dim, [h_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">nb_layers])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn_layers </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dim </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (in_dim, out_dim) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> zip</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)], layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    last_layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCNLayer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(layer_sizes[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dim; kwargs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dropout </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dropout)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; gcn_layers, dropout, last_layer) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, adj, mask)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> layer </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> gcn_layers</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> relu</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> dropout</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> last_layer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj))[:, mask]</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>GCN (generic function with 1 method)</span></span></code></pre></div><h2 id="Helper-Functions" tabindex="-1">Helper Functions <a class="header-anchor" href="#Helper-Functions" aria-label="Permalink to &quot;Helper Functions {#Helper-Functions}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y, adj, mask))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, adj, mask), ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; agg </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> mean, logits </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))(y_pred, y[:, mask])</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st, (; y_pred)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> onecold</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 100</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        hidden_dim</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, dropout</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, nb_layers</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, use_bias</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Bool</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        lr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.001</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, weight_decay</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Float64</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, patience</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 20</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, epochs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Int</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 200</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">default_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    Random</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">seed!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    features, targets, adj, (train_idx, val_idx, test_idx) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loadcora</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    gcn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> GCN</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(features, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), hidden_dim, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">size</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); nb_layers, dropout, use_bias)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, gcn) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> xdev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    opt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(weight_decay) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">?</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lr) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> AdamW</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lr, lambda </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> weight_decay)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(gcn, ps, st, opt)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Total Trainable Parameters: %0.4f M</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">parameterlength</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ps) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.0e6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_loss_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        gcn, ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st), (features, targets, adj, val_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, train_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    val_model_compiled </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compile</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((features, adj, val_idx), ps, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Inf</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">epochs</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            AutoEnzyme</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), loss_function, (features, targets, adj, train_idx), train_state;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            return_gradients </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        train_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                train_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, train_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, train_idx]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> first</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            val_loss_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                gcn, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                (features, targets, adj, val_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        val_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                val_model_compiled</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, val_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, val_idx]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch %3d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Train Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Val Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\\</span></span>
<span class="line"><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">                 Val Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss train_acc val_loss val_acc</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> best_loss_val</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            best_loss_val </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loss</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        else</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            if</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cnt </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> patience</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Early Stopping at Epoch %d</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                break</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">            end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        loss_function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            gcn, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (features, targets, adj, test_idx)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    test_acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @jit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                gcn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    (features, adj, test_idx),</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                    train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                )</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            )[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        ),</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        Array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(targets)[:, test_idx]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    )</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Test Loss: %.6f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\t</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">Test Acc: %.4f%%</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> test_loss test_acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2025-03-07 22:17:18.969573: I external/xla/xla/service/service.cc:152] XLA service 0xc7d60e0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:</span></span>
<span class="line"><span>2025-03-07 22:17:18.969794: I external/xla/xla/service/service.cc:160]   StreamExecutor device (0): NVIDIA A100-PCIE-40GB MIG 1g.5gb, Compute Capability 8.0</span></span>
<span class="line"><span>WARNING: All log messages before absl::InitializeLog() is called are written to STDERR</span></span>
<span class="line"><span>I0000 00:00:1741385838.971303  667263 se_gpu_pjrt_client.cc:951] Using BFC allocator.</span></span>
<span class="line"><span>I0000 00:00:1741385838.971425  667263 gpu_helpers.cc:136] XLA backend allocating 3825205248 bytes on device 0 for BFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741385838.971485  667263 gpu_helpers.cc:177] XLA backend will use up to 1275068416 bytes on device 0 for CollectiveBFCAllocator.</span></span>
<span class="line"><span>I0000 00:00:1741385838.988907  667263 cuda_dnn.cc:529] Loaded cuDNN version 90400</span></span>
<span class="line"><span>┌ Warning: \`replicate\` doesn&#39;t work for \`TaskLocalRNG\`. Returning the same \`TaskLocalRNG\`.</span></span>
<span class="line"><span>└ @ LuxCore /var/lib/buildkite-agent/builds/gpuci-2/julialang/lux-dot-jl/lib/LuxCore/src/LuxCore.jl:18</span></span>
<span class="line"><span>Total Trainable Parameters: 0.0964 M</span></span>
<span class="line"><span>┌ Warning: \`training\` is set to \`Val{false}()\` but is being used within an autodiff call (gradient, jacobian, etc...). This might lead to incorrect results. If you are using a \`Lux.jl\` model, set it to training mode using \`LuxCore.trainmode\`.</span></span>
<span class="line"><span>└ @ LuxLib.Utils /var/lib/buildkite-agent/builds/gpuci-2/julialang/lux-dot-jl/lib/LuxLib/src/utils.jl:340</span></span>
<span class="line"><span>2025-03-07 22:19:28.669951: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:28.722185: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 12 bytes spill stores, 12 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:28.746570: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 276 bytes spill stores, 276 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:28.868680: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 16 bytes spill stores, 16 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:28.974748: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 128 bytes spill stores, 128 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:29.143889: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22_0&#39;, 68 bytes spill stores, 68 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:29.188090: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 320 bytes spill stores, 320 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:29.247494: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_29&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:29.492471: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 336 bytes spill stores, 336 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:29.680553: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 1176 bytes spill stores, 1148 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:30.160208: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:30.418188: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_22&#39;, 648 bytes spill stores, 652 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:30.550147: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:31.179203: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_24&#39;, 304 bytes spill stores, 304 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:19:31.254111: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741385971.496325  667263 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741385971.496389  667263 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741385971.496394  667263 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741385971.496399  667263 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741385971.496403  667263 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741385971.496408  667263 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741385971.496413  667263 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741385971.496418  667263 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741385971.496422  667263 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741385971.496426  667263 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-03-07 22:19:31.496442: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.499542  667263 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741385971.499586  667263 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741385971.499591  667263 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741385971.499595  667263 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741385971.499599  667263 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741385971.499604  667263 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741385971.499608  667263 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741385971.499613  667263 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741385971.499617  667263 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741385971.499621  667263 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-03-07 22:19:31.499632: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.502287  667263 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741385971.502319  667263 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741385971.502324  667263 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741385971.502328  667263 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741385971.502333  667263 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741385971.502337  667263 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741385971.502341  667263 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741385971.502346  667263 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741385971.502350  667263 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741385971.502354  667263 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-03-07 22:19:31.502364: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.504904  667263 buffer_comparator.cc:156] Difference at 16: 0, expected 16.6863</span></span>
<span class="line"><span>E0000 00:00:1741385971.504933  667263 buffer_comparator.cc:156] Difference at 17: 0, expected 16.7136</span></span>
<span class="line"><span>E0000 00:00:1741385971.504936  667263 buffer_comparator.cc:156] Difference at 18: 0, expected 18.9638</span></span>
<span class="line"><span>E0000 00:00:1741385971.504939  667263 buffer_comparator.cc:156] Difference at 19: 0, expected 16.9232</span></span>
<span class="line"><span>E0000 00:00:1741385971.504942  667263 buffer_comparator.cc:156] Difference at 20: 0, expected 18.7426</span></span>
<span class="line"><span>E0000 00:00:1741385971.504945  667263 buffer_comparator.cc:156] Difference at 21: 0, expected 18.4411</span></span>
<span class="line"><span>E0000 00:00:1741385971.504948  667263 buffer_comparator.cc:156] Difference at 22: 0, expected 16.4673</span></span>
<span class="line"><span>E0000 00:00:1741385971.504951  667263 buffer_comparator.cc:156] Difference at 23: 0, expected 16.3965</span></span>
<span class="line"><span>E0000 00:00:1741385971.504953  667263 buffer_comparator.cc:156] Difference at 24: 0, expected 16.8089</span></span>
<span class="line"><span>E0000 00:00:1741385971.504956  667263 buffer_comparator.cc:156] Difference at 25: 0, expected 18.5767</span></span>
<span class="line"><span>2025-03-07 22:19:31.504965: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.507430  667263 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741385971.507454  667263 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741385971.507457  667263 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741385971.507460  667263 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741385971.507463  667263 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741385971.507466  667263 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741385971.507468  667263 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741385971.507471  667263 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741385971.507474  667263 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741385971.507477  667263 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-03-07 22:19:31.507484: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.509925  667263 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741385971.509943  667263 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741385971.509946  667263 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741385971.509949  667263 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741385971.509952  667263 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741385971.509955  667263 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741385971.509958  667263 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741385971.509961  667263 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741385971.509964  667263 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741385971.509967  667263 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-03-07 22:19:31.509973: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.512401  667263 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741385971.512420  667263 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741385971.512423  667263 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741385971.512426  667263 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741385971.512429  667263 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741385971.512432  667263 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741385971.512435  667263 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741385971.512437  667263 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741385971.512440  667263 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741385971.512443  667263 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-03-07 22:19:31.512448: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.514867  667263 buffer_comparator.cc:156] Difference at 32: 0, expected 16.6058</span></span>
<span class="line"><span>E0000 00:00:1741385971.514888  667263 buffer_comparator.cc:156] Difference at 33: 0, expected 16.5355</span></span>
<span class="line"><span>E0000 00:00:1741385971.514893  667263 buffer_comparator.cc:156] Difference at 34: 0, expected 15.8901</span></span>
<span class="line"><span>E0000 00:00:1741385971.514896  667263 buffer_comparator.cc:156] Difference at 35: 0, expected 17.0544</span></span>
<span class="line"><span>E0000 00:00:1741385971.514899  667263 buffer_comparator.cc:156] Difference at 36: 0, expected 19.5872</span></span>
<span class="line"><span>E0000 00:00:1741385971.514902  667263 buffer_comparator.cc:156] Difference at 37: 0, expected 18.6532</span></span>
<span class="line"><span>E0000 00:00:1741385971.514905  667263 buffer_comparator.cc:156] Difference at 38: 0, expected 17.0428</span></span>
<span class="line"><span>E0000 00:00:1741385971.514907  667263 buffer_comparator.cc:156] Difference at 39: 0, expected 16.5588</span></span>
<span class="line"><span>E0000 00:00:1741385971.514910  667263 buffer_comparator.cc:156] Difference at 40: 0, expected 16.588</span></span>
<span class="line"><span>E0000 00:00:1741385971.514913  667263 buffer_comparator.cc:156] Difference at 41: 0, expected 20.3484</span></span>
<span class="line"><span>2025-03-07 22:19:31.514919: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.517376  667263 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741385971.517408  667263 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741385971.517411  667263 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741385971.517414  667263 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741385971.517417  667263 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741385971.517420  667263 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741385971.517423  667263 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741385971.517426  667263 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741385971.517429  667263 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741385971.517431  667263 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-07 22:19:31.517438: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.519927  667263 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741385971.519951  667263 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741385971.519954  667263 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741385971.519957  667263 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741385971.519960  667263 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741385971.519963  667263 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741385971.519966  667263 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741385971.519969  667263 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741385971.519972  667263 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741385971.519975  667263 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-07 22:19:31.519980: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.522451  667263 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741385971.522477  667263 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741385971.522481  667263 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741385971.522484  667263 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741385971.522487  667263 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741385971.522489  667263 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741385971.522493  667263 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741385971.522496  667263 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741385971.522499  667263 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741385971.522502  667263 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-07 22:19:31.522509: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.524992  667263 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741385971.525012  667263 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741385971.525015  667263 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741385971.525018  667263 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741385971.525021  667263 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741385971.525024  667263 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741385971.525027  667263 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741385971.525030  667263 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741385971.525033  667263 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741385971.525036  667263 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-07 22:19:31.525042: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.527476  667263 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741385971.527493  667263 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741385971.527496  667263 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741385971.527500  667263 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741385971.527502  667263 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741385971.527505  667263 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741385971.527508  667263 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741385971.527511  667263 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741385971.527514  667263 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741385971.527517  667263 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-07 22:19:31.527521: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.529940  667263 buffer_comparator.cc:156] Difference at 64: 0, expected 17.5724</span></span>
<span class="line"><span>E0000 00:00:1741385971.529962  667263 buffer_comparator.cc:156] Difference at 65: 0, expected 18.923</span></span>
<span class="line"><span>E0000 00:00:1741385971.529966  667263 buffer_comparator.cc:156] Difference at 66: 0, expected 18.4123</span></span>
<span class="line"><span>E0000 00:00:1741385971.529969  667263 buffer_comparator.cc:156] Difference at 67: 0, expected 17.9799</span></span>
<span class="line"><span>E0000 00:00:1741385971.529971  667263 buffer_comparator.cc:156] Difference at 68: 0, expected 17.9802</span></span>
<span class="line"><span>E0000 00:00:1741385971.529974  667263 buffer_comparator.cc:156] Difference at 69: 0, expected 19.1597</span></span>
<span class="line"><span>E0000 00:00:1741385971.529977  667263 buffer_comparator.cc:156] Difference at 70: 0, expected 17.2382</span></span>
<span class="line"><span>E0000 00:00:1741385971.529980  667263 buffer_comparator.cc:156] Difference at 71: 0, expected 19.1951</span></span>
<span class="line"><span>E0000 00:00:1741385971.529983  667263 buffer_comparator.cc:156] Difference at 72: 0, expected 15.634</span></span>
<span class="line"><span>E0000 00:00:1741385971.529986  667263 buffer_comparator.cc:156] Difference at 73: 0, expected 17.8359</span></span>
<span class="line"><span>2025-03-07 22:19:31.530003: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.532437  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1741385971.532458  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1741385971.532461  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1741385971.532464  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1741385971.532467  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1741385971.532470  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1741385971.532473  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1741385971.532475  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1741385971.532478  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1741385971.532481  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-03-07 22:19:31.532486: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.534934  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1741385971.534953  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1741385971.534956  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1741385971.534959  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1741385971.534961  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1741385971.534964  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1741385971.534967  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1741385971.534970  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1741385971.534973  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1741385971.534976  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-03-07 22:19:31.534980: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.537433  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1741385971.537453  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1741385971.537456  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1741385971.537459  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1741385971.537462  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1741385971.537465  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1741385971.537468  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1741385971.537470  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1741385971.537473  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1741385971.537476  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-03-07 22:19:31.537482: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.540007  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 22.1715</span></span>
<span class="line"><span>E0000 00:00:1741385971.540028  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 14.8073</span></span>
<span class="line"><span>E0000 00:00:1741385971.540032  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 17.4052</span></span>
<span class="line"><span>E0000 00:00:1741385971.540035  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 19.8808</span></span>
<span class="line"><span>E0000 00:00:1741385971.540038  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 18.8043</span></span>
<span class="line"><span>E0000 00:00:1741385971.540041  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 18.2218</span></span>
<span class="line"><span>E0000 00:00:1741385971.540044  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 18.3321</span></span>
<span class="line"><span>E0000 00:00:1741385971.540047  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 17.9272</span></span>
<span class="line"><span>E0000 00:00:1741385971.540050  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 15.649</span></span>
<span class="line"><span>E0000 00:00:1741385971.540052  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 18.5916</span></span>
<span class="line"><span>2025-03-07 22:19:31.540058: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.542518  667263 buffer_comparator.cc:156] Difference at 256: 0, expected 16.0393</span></span>
<span class="line"><span>E0000 00:00:1741385971.542539  667263 buffer_comparator.cc:156] Difference at 257: 0, expected 18.4933</span></span>
<span class="line"><span>E0000 00:00:1741385971.542542  667263 buffer_comparator.cc:156] Difference at 258: 0, expected 18.027</span></span>
<span class="line"><span>E0000 00:00:1741385971.542545  667263 buffer_comparator.cc:156] Difference at 259: 0, expected 20.7645</span></span>
<span class="line"><span>E0000 00:00:1741385971.542548  667263 buffer_comparator.cc:156] Difference at 260: 0, expected 18.8066</span></span>
<span class="line"><span>E0000 00:00:1741385971.542551  667263 buffer_comparator.cc:156] Difference at 261: 0, expected 17.9486</span></span>
<span class="line"><span>E0000 00:00:1741385971.542553  667263 buffer_comparator.cc:156] Difference at 262: 0, expected 16.8675</span></span>
<span class="line"><span>E0000 00:00:1741385971.542556  667263 buffer_comparator.cc:156] Difference at 263: 0, expected 18.7938</span></span>
<span class="line"><span>E0000 00:00:1741385971.542559  667263 buffer_comparator.cc:156] Difference at 264: 0, expected 16.5109</span></span>
<span class="line"><span>E0000 00:00:1741385971.542562  667263 buffer_comparator.cc:156] Difference at 265: 0, expected 20.2758</span></span>
<span class="line"><span>2025-03-07 22:19:31.542567: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.545004  667263 buffer_comparator.cc:156] Difference at 256: 0, expected 16.0393</span></span>
<span class="line"><span>E0000 00:00:1741385971.545029  667263 buffer_comparator.cc:156] Difference at 257: 0, expected 18.4933</span></span>
<span class="line"><span>E0000 00:00:1741385971.545032  667263 buffer_comparator.cc:156] Difference at 258: 0, expected 18.027</span></span>
<span class="line"><span>E0000 00:00:1741385971.545035  667263 buffer_comparator.cc:156] Difference at 259: 0, expected 20.7645</span></span>
<span class="line"><span>E0000 00:00:1741385971.545038  667263 buffer_comparator.cc:156] Difference at 260: 0, expected 18.8066</span></span>
<span class="line"><span>E0000 00:00:1741385971.545041  667263 buffer_comparator.cc:156] Difference at 261: 0, expected 17.9486</span></span>
<span class="line"><span>E0000 00:00:1741385971.545044  667263 buffer_comparator.cc:156] Difference at 262: 0, expected 16.8675</span></span>
<span class="line"><span>E0000 00:00:1741385971.545047  667263 buffer_comparator.cc:156] Difference at 263: 0, expected 18.7938</span></span>
<span class="line"><span>E0000 00:00:1741385971.545049  667263 buffer_comparator.cc:156] Difference at 264: 0, expected 16.5109</span></span>
<span class="line"><span>E0000 00:00:1741385971.545052  667263 buffer_comparator.cc:156] Difference at 265: 0, expected 20.2758</span></span>
<span class="line"><span>2025-03-07 22:19:31.545059: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.553174  667263 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1741385971.553207  667263 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1741385971.553210  667263 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1741385971.553213  667263 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1741385971.553216  667263 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1741385971.553219  667263 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1741385971.553223  667263 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1741385971.553226  667263 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1741385971.553229  667263 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.553232  667263 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-07 22:19:31.553242: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.559157  667263 buffer_comparator.cc:156] Difference at 16: 0, expected 1111.43</span></span>
<span class="line"><span>E0000 00:00:1741385971.559203  667263 buffer_comparator.cc:156] Difference at 17: 0, expected 1083.84</span></span>
<span class="line"><span>E0000 00:00:1741385971.559206  667263 buffer_comparator.cc:156] Difference at 18: 0, expected 1092.57</span></span>
<span class="line"><span>E0000 00:00:1741385971.559209  667263 buffer_comparator.cc:156] Difference at 19: 0, expected 1118.75</span></span>
<span class="line"><span>E0000 00:00:1741385971.559212  667263 buffer_comparator.cc:156] Difference at 20: 0, expected 1088.49</span></span>
<span class="line"><span>E0000 00:00:1741385971.559215  667263 buffer_comparator.cc:156] Difference at 21: 0, expected 1083.52</span></span>
<span class="line"><span>E0000 00:00:1741385971.559218  667263 buffer_comparator.cc:156] Difference at 22: 0, expected 1097.64</span></span>
<span class="line"><span>E0000 00:00:1741385971.559221  667263 buffer_comparator.cc:156] Difference at 23: 0, expected 1122.75</span></span>
<span class="line"><span>E0000 00:00:1741385971.559224  667263 buffer_comparator.cc:156] Difference at 24: 0, expected 1084.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.559227  667263 buffer_comparator.cc:156] Difference at 25: 0, expected 1084.58</span></span>
<span class="line"><span>2025-03-07 22:19:31.559237: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.564247  667263 buffer_comparator.cc:156] Difference at 64: 0, expected 1106.21</span></span>
<span class="line"><span>E0000 00:00:1741385971.564299  667263 buffer_comparator.cc:156] Difference at 65: 0, expected 1087.83</span></span>
<span class="line"><span>E0000 00:00:1741385971.564302  667263 buffer_comparator.cc:156] Difference at 66: 0, expected 1090.54</span></span>
<span class="line"><span>E0000 00:00:1741385971.564305  667263 buffer_comparator.cc:156] Difference at 67: 0, expected 1104.23</span></span>
<span class="line"><span>E0000 00:00:1741385971.564308  667263 buffer_comparator.cc:156] Difference at 68: 0, expected 1104.3</span></span>
<span class="line"><span>E0000 00:00:1741385971.564311  667263 buffer_comparator.cc:156] Difference at 69: 0, expected 1093.45</span></span>
<span class="line"><span>E0000 00:00:1741385971.564314  667263 buffer_comparator.cc:156] Difference at 70: 0, expected 1091.52</span></span>
<span class="line"><span>E0000 00:00:1741385971.564317  667263 buffer_comparator.cc:156] Difference at 71: 0, expected 1110.4</span></span>
<span class="line"><span>E0000 00:00:1741385971.564320  667263 buffer_comparator.cc:156] Difference at 72: 0, expected 1106.92</span></span>
<span class="line"><span>E0000 00:00:1741385971.564323  667263 buffer_comparator.cc:156] Difference at 73: 0, expected 1088.44</span></span>
<span class="line"><span>2025-03-07 22:19:31.564333: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.569201  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.569242  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.569245  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.569248  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.569251  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.569254  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.569258  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.569260  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.569263  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.569266  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.569277: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.574221  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.574261  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.574264  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.574267  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.574270  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.574273  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.574276  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.574279  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.574282  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.574285  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.574294: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.578892  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.578931  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.578935  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.578938  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.578941  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.578944  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.578946  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.578949  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.578952  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.578955  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.578964: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.583445  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.583479  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.583482  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.583485  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.583488  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.583491  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.583494  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.583497  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.583500  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.583503  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.583511: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.588077  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.588118  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.588124  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.588127  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.588130  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.588133  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.588136  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.588139  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.588142  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.588145  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.588153: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.592870  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.592909  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.592912  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.592915  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.592918  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.592921  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.592924  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.592927  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.592930  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.592933  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.592943: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.597527  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.597563  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.597566  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.597569  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.597572  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.597575  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.597578  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.597581  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.597584  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.597586  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.597595: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.602095  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.602135  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.602139  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.602142  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.602145  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.602147  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.602152  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.602155  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.602158  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.602161  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.602170: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.606888  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.606931  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.606935  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.606938  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.606940  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.606943  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.606946  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.606949  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.606952  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.606955  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.606964: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.611417  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.611468  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.611471  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.611474  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.611477  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.611480  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.611483  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.611486  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.611489  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.611492  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.611502: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.615942  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.615983  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.615986  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.615989  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.615992  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.615995  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.615998  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.616001  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.616004  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.616007  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.616018: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.620333  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.620379  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.620382  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.620385  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.620388  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.620391  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.620394  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.620397  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.620400  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.620403  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.620411: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.624942  667263 buffer_comparator.cc:156] Difference at 128: 0, expected 1099.62</span></span>
<span class="line"><span>E0000 00:00:1741385971.624969  667263 buffer_comparator.cc:156] Difference at 129: 0, expected 1084.13</span></span>
<span class="line"><span>E0000 00:00:1741385971.624972  667263 buffer_comparator.cc:156] Difference at 130: 0, expected 1112.78</span></span>
<span class="line"><span>E0000 00:00:1741385971.624975  667263 buffer_comparator.cc:156] Difference at 131: 0, expected 1094.63</span></span>
<span class="line"><span>E0000 00:00:1741385971.624978  667263 buffer_comparator.cc:156] Difference at 132: 0, expected 1088.07</span></span>
<span class="line"><span>E0000 00:00:1741385971.624981  667263 buffer_comparator.cc:156] Difference at 133: 0, expected 1107.65</span></span>
<span class="line"><span>E0000 00:00:1741385971.624984  667263 buffer_comparator.cc:156] Difference at 134: 0, expected 1101.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.624987  667263 buffer_comparator.cc:156] Difference at 135: 0, expected 1110.56</span></span>
<span class="line"><span>E0000 00:00:1741385971.624990  667263 buffer_comparator.cc:156] Difference at 136: 0, expected 1080.81</span></span>
<span class="line"><span>E0000 00:00:1741385971.624993  667263 buffer_comparator.cc:156] Difference at 137: 0, expected 1091.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.625000: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.629362  667263 buffer_comparator.cc:156] Difference at 256: 0, expected 1091.26</span></span>
<span class="line"><span>E0000 00:00:1741385971.629387  667263 buffer_comparator.cc:156] Difference at 257: 0, expected 1117.91</span></span>
<span class="line"><span>E0000 00:00:1741385971.629390  667263 buffer_comparator.cc:156] Difference at 258: 0, expected 1086.11</span></span>
<span class="line"><span>E0000 00:00:1741385971.629393  667263 buffer_comparator.cc:156] Difference at 259: 0, expected 1095.59</span></span>
<span class="line"><span>E0000 00:00:1741385971.629396  667263 buffer_comparator.cc:156] Difference at 260: 0, expected 1098.42</span></span>
<span class="line"><span>E0000 00:00:1741385971.629399  667263 buffer_comparator.cc:156] Difference at 261: 0, expected 1113.28</span></span>
<span class="line"><span>E0000 00:00:1741385971.629402  667263 buffer_comparator.cc:156] Difference at 262: 0, expected 1088.03</span></span>
<span class="line"><span>E0000 00:00:1741385971.629405  667263 buffer_comparator.cc:156] Difference at 263: 0, expected 1093.88</span></span>
<span class="line"><span>E0000 00:00:1741385971.629408  667263 buffer_comparator.cc:156] Difference at 264: 0, expected 1115.18</span></span>
<span class="line"><span>E0000 00:00:1741385971.629411  667263 buffer_comparator.cc:156] Difference at 265: 0, expected 1104.89</span></span>
<span class="line"><span>2025-03-07 22:19:31.629417: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.640416  667263 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741385971.640463  667263 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741385971.640468  667263 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741385971.640471  667263 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741385971.640474  667263 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741385971.640477  667263 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741385971.640480  667263 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741385971.640483  667263 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741385971.640486  667263 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741385971.640489  667263 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.640500: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.643516  667263 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741385971.643534  667263 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741385971.643537  667263 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741385971.643540  667263 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741385971.643543  667263 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741385971.643546  667263 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741385971.643549  667263 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741385971.643552  667263 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741385971.643555  667263 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741385971.643558  667263 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.643564: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.646576  667263 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741385971.646593  667263 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741385971.646596  667263 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741385971.646599  667263 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741385971.646603  667263 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741385971.646606  667263 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741385971.646609  667263 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741385971.646612  667263 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741385971.646615  667263 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741385971.646618  667263 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.646623: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.649620  667263 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741385971.649635  667263 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741385971.649639  667263 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741385971.649642  667263 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741385971.649647  667263 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741385971.649650  667263 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741385971.649653  667263 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741385971.649656  667263 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741385971.649659  667263 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741385971.649662  667263 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.649667: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.652737  667263 buffer_comparator.cc:156] Difference at 0: 1139.71, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1741385971.652751  667263 buffer_comparator.cc:156] Difference at 1: 1404.8, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1741385971.652755  667263 buffer_comparator.cc:156] Difference at 2: 2132.23, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1741385971.652758  667263 buffer_comparator.cc:156] Difference at 3: 1838.84, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1741385971.652761  667263 buffer_comparator.cc:156] Difference at 4: 1307.39, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1741385971.652764  667263 buffer_comparator.cc:156] Difference at 5: 2064.39, expected 1757.79</span></span>
<span class="line"><span>E0000 00:00:1741385971.652767  667263 buffer_comparator.cc:156] Difference at 6: 1480.82, expected 1259.42</span></span>
<span class="line"><span>E0000 00:00:1741385971.652770  667263 buffer_comparator.cc:156] Difference at 7: 1113.19, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1741385971.652773  667263 buffer_comparator.cc:156] Difference at 8: 1358.65, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.652776  667263 buffer_comparator.cc:156] Difference at 9: 2048.24, expected 1833.76</span></span>
<span class="line"><span>2025-03-07 22:19:31.652781: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.655833  667263 buffer_comparator.cc:156] Difference at 112: 1196.02, expected 949.79</span></span>
<span class="line"><span>E0000 00:00:1741385971.655847  667263 buffer_comparator.cc:156] Difference at 113: 1042.17, expected 1213.3</span></span>
<span class="line"><span>E0000 00:00:1741385971.655851  667263 buffer_comparator.cc:156] Difference at 114: 726.264, expected 1837.44</span></span>
<span class="line"><span>E0000 00:00:1741385971.655854  667263 buffer_comparator.cc:156] Difference at 115: 1164.44, expected 1600.27</span></span>
<span class="line"><span>E0000 00:00:1741385971.655857  667263 buffer_comparator.cc:156] Difference at 116: 838.315, expected 1117.2</span></span>
<span class="line"><span>E0000 00:00:1741385971.655860  667263 buffer_comparator.cc:156] Difference at 117: 618.979, expected 1790.83</span></span>
<span class="line"><span>E0000 00:00:1741385971.655863  667263 buffer_comparator.cc:156] Difference at 118: 782.852, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741385971.655866  667263 buffer_comparator.cc:156] Difference at 119: 1182.07, expected 943.242</span></span>
<span class="line"><span>E0000 00:00:1741385971.655869  667263 buffer_comparator.cc:156] Difference at 120: 1033.7, expected 1198.86</span></span>
<span class="line"><span>E0000 00:00:1741385971.655872  667263 buffer_comparator.cc:156] Difference at 121: 728.147, expected 1820.15</span></span>
<span class="line"><span>2025-03-07 22:19:31.655877: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.658931  667263 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741385971.658946  667263 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741385971.658949  667263 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741385971.658953  667263 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741385971.658956  667263 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741385971.658959  667263 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741385971.658964  667263 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741385971.658967  667263 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741385971.658970  667263 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741385971.658973  667263 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-07 22:19:31.658978: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.661958  667263 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741385971.661976  667263 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741385971.661979  667263 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741385971.661982  667263 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741385971.661985  667263 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741385971.661988  667263 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741385971.661991  667263 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741385971.661994  667263 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741385971.661997  667263 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741385971.662000  667263 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-07 22:19:31.662005: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.665002  667263 buffer_comparator.cc:156] Difference at 224: 1186.14, expected 942.345</span></span>
<span class="line"><span>E0000 00:00:1741385971.665019  667263 buffer_comparator.cc:156] Difference at 225: 1033.68, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741385971.665022  667263 buffer_comparator.cc:156] Difference at 226: 723.67, expected 1824.94</span></span>
<span class="line"><span>E0000 00:00:1741385971.665025  667263 buffer_comparator.cc:156] Difference at 227: 1156.29, expected 1592.15</span></span>
<span class="line"><span>E0000 00:00:1741385971.665028  667263 buffer_comparator.cc:156] Difference at 228: 843.86, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741385971.665031  667263 buffer_comparator.cc:156] Difference at 229: 633.168, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741385971.665034  667263 buffer_comparator.cc:156] Difference at 230: 810.302, expected 1283.28</span></span>
<span class="line"><span>E0000 00:00:1741385971.665038  667263 buffer_comparator.cc:156] Difference at 231: 1218.15, expected 935.373</span></span>
<span class="line"><span>E0000 00:00:1741385971.665041  667263 buffer_comparator.cc:156] Difference at 232: 1064.04, expected 1192.72</span></span>
<span class="line"><span>E0000 00:00:1741385971.665044  667263 buffer_comparator.cc:156] Difference at 233: 741.156, expected 1803.13</span></span>
<span class="line"><span>2025-03-07 22:19:31.665049: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.668149  667263 buffer_comparator.cc:156] Difference at 448: 1214.22, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741385971.668163  667263 buffer_comparator.cc:156] Difference at 449: 1056.45, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741385971.668167  667263 buffer_comparator.cc:156] Difference at 450: 736.847, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741385971.668170  667263 buffer_comparator.cc:156] Difference at 451: 1184.91, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741385971.668173  667263 buffer_comparator.cc:156] Difference at 452: 859.942, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741385971.668176  667263 buffer_comparator.cc:156] Difference at 453: 620.77, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741385971.668179  667263 buffer_comparator.cc:156] Difference at 454: 796.75, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741385971.668182  667263 buffer_comparator.cc:156] Difference at 455: 1201.02, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741385971.668186  667263 buffer_comparator.cc:156] Difference at 456: 1045.45, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741385971.668189  667263 buffer_comparator.cc:156] Difference at 457: 732.834, expected 1821.28</span></span>
<span class="line"><span>2025-03-07 22:19:31.668194: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.671174  667263 buffer_comparator.cc:156] Difference at 0: 1057.27, expected 928.593</span></span>
<span class="line"><span>E0000 00:00:1741385971.671190  667263 buffer_comparator.cc:156] Difference at 1: 1319.15, expected 1186.89</span></span>
<span class="line"><span>E0000 00:00:1741385971.671193  667263 buffer_comparator.cc:156] Difference at 2: 2004.43, expected 1796.77</span></span>
<span class="line"><span>E0000 00:00:1741385971.671196  667263 buffer_comparator.cc:156] Difference at 3: 1745.74, expected 1565.84</span></span>
<span class="line"><span>E0000 00:00:1741385971.671199  667263 buffer_comparator.cc:156] Difference at 4: 1252.2, expected 1095.49</span></span>
<span class="line"><span>E0000 00:00:1741385971.671202  667263 buffer_comparator.cc:156] Difference at 7: 1175.57, expected 951.95</span></span>
<span class="line"><span>E0000 00:00:1741385971.671205  667263 buffer_comparator.cc:156] Difference at 8: 1398.75, expected 1216.24</span></span>
<span class="line"><span>E0000 00:00:1741385971.671208  667263 buffer_comparator.cc:156] Difference at 9: 2125.62, expected 1833.76</span></span>
<span class="line"><span>E0000 00:00:1741385971.671211  667263 buffer_comparator.cc:156] Difference at 10: 1878.38, expected 1592.37</span></span>
<span class="line"><span>E0000 00:00:1741385971.671214  667263 buffer_comparator.cc:156] Difference at 11: 1362.67, expected 1121.95</span></span>
<span class="line"><span>2025-03-07 22:19:31.671219: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.674249  667263 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741385971.674267  667263 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741385971.674270  667263 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741385971.674273  667263 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741385971.674276  667263 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741385971.674279  667263 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741385971.674282  667263 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741385971.674285  667263 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741385971.674288  667263 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741385971.674291  667263 buffer_comparator.cc:156] Difference at 457: 737.437, expected 1821.28</span></span>
<span class="line"><span>2025-03-07 22:19:31.674296: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741385971.677323  667263 buffer_comparator.cc:156] Difference at 448: 1221.14, expected 948.676</span></span>
<span class="line"><span>E0000 00:00:1741385971.677340  667263 buffer_comparator.cc:156] Difference at 449: 1061.5, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741385971.677343  667263 buffer_comparator.cc:156] Difference at 450: 743.315, expected 1813.49</span></span>
<span class="line"><span>E0000 00:00:1741385971.677346  667263 buffer_comparator.cc:156] Difference at 451: 1192.79, expected 1575.23</span></span>
<span class="line"><span>E0000 00:00:1741385971.677349  667263 buffer_comparator.cc:156] Difference at 452: 864.899, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741385971.677352  667263 buffer_comparator.cc:156] Difference at 453: 626.203, expected 1764.87</span></span>
<span class="line"><span>E0000 00:00:1741385971.677355  667263 buffer_comparator.cc:156] Difference at 454: 803.97, expected 1269.69</span></span>
<span class="line"><span>E0000 00:00:1741385971.677358  667263 buffer_comparator.cc:156] Difference at 455: 1208.29, expected 952.925</span></span>
<span class="line"><span>E0000 00:00:1741385971.677361  667263 buffer_comparator.cc:156] Difference at 456: 1052.01, expected 1213.59</span></span>
<span class="line"><span>Epoch   1	Train Loss: 14.779691	Train Acc: 22.1429%	Val Loss: 7.846817	Val Acc: 25.8000%</span></span>
<span class="line"><span>Epoch   2	Train Loss: 7.980669	Train Acc: 22.8571%	Val Loss: 4.318908	Val Acc: 29.0000%</span></span>
<span class="line"><span>Epoch   3	Train Loss: 3.608830	Train Acc: 40.7143%	Val Loss: 2.339689	Val Acc: 34.6000%</span></span>
<span class="line"><span>Epoch   4	Train Loss: 2.165077	Train Acc: 56.4286%	Val Loss: 1.921515	Val Acc: 44.8000%</span></span>
<span class="line"><span>Epoch   5	Train Loss: 1.673108	Train Acc: 63.5714%	Val Loss: 1.949382	Val Acc: 46.2000%</span></span>
<span class="line"><span>Epoch   6	Train Loss: 1.473504	Train Acc: 70.0000%	Val Loss: 1.929327	Val Acc: 47.4000%</span></span>
<span class="line"><span>Epoch   7	Train Loss: 1.378564	Train Acc: 75.7143%	Val Loss: 1.815769	Val Acc: 52.4000%</span></span>
<span class="line"><span>Epoch   8	Train Loss: 1.266600	Train Acc: 77.8571%	Val Loss: 1.690460	Val Acc: 58.2000%</span></span>
<span class="line"><span>Epoch   9	Train Loss: 1.131189	Train Acc: 79.2857%	Val Loss: 1.590804	Val Acc: 60.2000%</span></span>
<span class="line"><span>Epoch  10	Train Loss: 1.009956	Train Acc: 79.2857%	Val Loss: 1.537189	Val Acc: 62.2000%</span></span>
<span class="line"><span>Epoch  11	Train Loss: 0.909779	Train Acc: 80.0000%	Val Loss: 1.522853	Val Acc: 62.8000%</span></span>
<span class="line"><span>Epoch  12	Train Loss: 0.835255	Train Acc: 78.5714%	Val Loss: 1.524822	Val Acc: 62.6000%</span></span>
<span class="line"><span>Epoch  13	Train Loss: 0.763603	Train Acc: 80.0000%	Val Loss: 1.538342	Val Acc: 63.0000%</span></span>
<span class="line"><span>Epoch  14	Train Loss: 0.729771	Train Acc: 80.0000%	Val Loss: 1.555343	Val Acc: 63.8000%</span></span>
<span class="line"><span>Epoch  15	Train Loss: 0.730870	Train Acc: 80.7143%	Val Loss: 1.563835	Val Acc: 63.0000%</span></span>
<span class="line"><span>Epoch  16	Train Loss: 0.691014	Train Acc: 82.8571%	Val Loss: 1.565136	Val Acc: 63.2000%</span></span>
<span class="line"><span>Epoch  17	Train Loss: 0.611773	Train Acc: 83.5714%	Val Loss: 1.583633	Val Acc: 61.8000%</span></span>
<span class="line"><span>Epoch  18	Train Loss: 0.543218	Train Acc: 84.2857%	Val Loss: 1.624722	Val Acc: 61.2000%</span></span>
<span class="line"><span>Epoch  19	Train Loss: 0.512904	Train Acc: 85.0000%	Val Loss: 1.676172	Val Acc: 61.0000%</span></span>
<span class="line"><span>Epoch  20	Train Loss: 0.501060	Train Acc: 85.0000%	Val Loss: 1.715239	Val Acc: 62.0000%</span></span>
<span class="line"><span>Epoch  21	Train Loss: 0.493121	Train Acc: 85.7143%	Val Loss: 1.733262	Val Acc: 62.2000%</span></span>
<span class="line"><span>Epoch  22	Train Loss: 0.478811	Train Acc: 86.4286%	Val Loss: 1.731615	Val Acc: 62.8000%</span></span>
<span class="line"><span>Epoch  23	Train Loss: 0.456186	Train Acc: 86.4286%	Val Loss: 1.717054	Val Acc: 63.0000%</span></span>
<span class="line"><span>Epoch  24	Train Loss: 0.428612	Train Acc: 86.4286%	Val Loss: 1.696958	Val Acc: 63.6000%</span></span>
<span class="line"><span>Epoch  25	Train Loss: 0.402466	Train Acc: 87.8571%	Val Loss: 1.679504	Val Acc: 64.6000%</span></span>
<span class="line"><span>Epoch  26	Train Loss: 0.385548	Train Acc: 89.2857%	Val Loss: 1.668218	Val Acc: 65.6000%</span></span>
<span class="line"><span>Epoch  27	Train Loss: 0.374205	Train Acc: 90.7143%	Val Loss: 1.661685	Val Acc: 65.8000%</span></span>
<span class="line"><span>Epoch  28	Train Loss: 0.365152	Train Acc: 92.1429%	Val Loss: 1.659513	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  29	Train Loss: 0.357253	Train Acc: 92.1429%	Val Loss: 1.660682	Val Acc: 66.4000%</span></span>
<span class="line"><span>Epoch  30	Train Loss: 0.349820	Train Acc: 92.1429%	Val Loss: 1.665978	Val Acc: 66.0000%</span></span>
<span class="line"><span>Epoch  31	Train Loss: 0.342369	Train Acc: 92.1429%	Val Loss: 1.674809	Val Acc: 66.0000%</span></span>
<span class="line"><span>Early Stopping at Epoch 31</span></span>
<span class="line"><span>2025-03-07 22:20:14.177464: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:20:14.277130: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:20:14.298789: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_33_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741386014.427243  667263 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741386014.427316  667263 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741386014.427325  667263 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741386014.427332  667263 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.427340  667263 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741386014.427347  667263 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741386014.427354  667263 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.427361  667263 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741386014.427368  667263 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.427375  667263 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-07 22:20:14.427392: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.430870  667263 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741386014.430913  667263 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741386014.430916  667263 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741386014.430920  667263 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.430923  667263 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741386014.430926  667263 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741386014.430929  667263 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.430932  667263 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741386014.430935  667263 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.430938  667263 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-07 22:20:14.430950: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.434337  667263 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741386014.434363  667263 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741386014.434366  667263 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741386014.434369  667263 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.434372  667263 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741386014.434376  667263 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741386014.434379  667263 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.434382  667263 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741386014.434387  667263 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.434390  667263 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-07 22:20:14.434397: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.437681  667263 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741386014.437711  667263 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741386014.437714  667263 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741386014.437718  667263 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.437721  667263 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741386014.437724  667263 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741386014.437727  667263 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.437730  667263 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741386014.437733  667263 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.437736  667263 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-07 22:20:14.437744: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.441051  667263 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741386014.441069  667263 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741386014.441072  667263 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741386014.441075  667263 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.441078  667263 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741386014.441082  667263 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741386014.441085  667263 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.441088  667263 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741386014.441091  667263 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.441094  667263 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-07 22:20:14.441101: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.444393  667263 buffer_comparator.cc:156] Difference at 112: 491.274, expected 949.794</span></span>
<span class="line"><span>E0000 00:00:1741386014.444416  667263 buffer_comparator.cc:156] Difference at 113: 739.764, expected 1213.31</span></span>
<span class="line"><span>E0000 00:00:1741386014.444419  667263 buffer_comparator.cc:156] Difference at 114: 647.819, expected 1837.45</span></span>
<span class="line"><span>E0000 00:00:1741386014.444423  667263 buffer_comparator.cc:156] Difference at 115: 452.429, expected 1600.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.444426  667263 buffer_comparator.cc:156] Difference at 116: 720.609, expected 1117.21</span></span>
<span class="line"><span>E0000 00:00:1741386014.444429  667263 buffer_comparator.cc:156] Difference at 117: 520.117, expected 1790.84</span></span>
<span class="line"><span>E0000 00:00:1741386014.444432  667263 buffer_comparator.cc:156] Difference at 118: 366.392, expected 1291.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.444435  667263 buffer_comparator.cc:156] Difference at 119: 475.884, expected 943.247</span></span>
<span class="line"><span>E0000 00:00:1741386014.444438  667263 buffer_comparator.cc:156] Difference at 120: 710.718, expected 1198.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.444441  667263 buffer_comparator.cc:156] Difference at 121: 624.258, expected 1820.16</span></span>
<span class="line"><span>2025-03-07 22:20:14.444450: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.447707  667263 buffer_comparator.cc:156] Difference at 224: 474.076, expected 942.349</span></span>
<span class="line"><span>E0000 00:00:1741386014.447730  667263 buffer_comparator.cc:156] Difference at 225: 706.778, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741386014.447734  667263 buffer_comparator.cc:156] Difference at 226: 619.638, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741386014.447737  667263 buffer_comparator.cc:156] Difference at 227: 429.558, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741386014.447740  667263 buffer_comparator.cc:156] Difference at 228: 687.576, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741386014.447743  667263 buffer_comparator.cc:156] Difference at 229: 493.772, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741386014.447746  667263 buffer_comparator.cc:156] Difference at 230: 364.686, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741386014.447749  667263 buffer_comparator.cc:156] Difference at 231: 476.353, expected 935.377</span></span>
<span class="line"><span>E0000 00:00:1741386014.447752  667263 buffer_comparator.cc:156] Difference at 232: 716.397, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741386014.447755  667263 buffer_comparator.cc:156] Difference at 233: 619.223, expected 1803.14</span></span>
<span class="line"><span>2025-03-07 22:20:14.447763: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.450968  667263 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741386014.450997  667263 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741386014.451001  667263 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741386014.451004  667263 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741386014.451007  667263 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741386014.451010  667263 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741386014.451013  667263 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741386014.451016  667263 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1741386014.451019  667263 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1741386014.451022  667263 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-07 22:20:14.451031: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.454337  667263 buffer_comparator.cc:156] Difference at 225: 1429.96, expected 1208.53</span></span>
<span class="line"><span>E0000 00:00:1741386014.454366  667263 buffer_comparator.cc:156] Difference at 226: 1251.98, expected 1824.95</span></span>
<span class="line"><span>E0000 00:00:1741386014.454369  667263 buffer_comparator.cc:156] Difference at 227: 872.927, expected 1592.16</span></span>
<span class="line"><span>E0000 00:00:1741386014.454372  667263 buffer_comparator.cc:156] Difference at 228: 1392.16, expected 1119.85</span></span>
<span class="line"><span>E0000 00:00:1741386014.454375  667263 buffer_comparator.cc:156] Difference at 229: 994.073, expected 1778.8</span></span>
<span class="line"><span>E0000 00:00:1741386014.454378  667263 buffer_comparator.cc:156] Difference at 230: 754.206, expected 1283.29</span></span>
<span class="line"><span>E0000 00:00:1741386014.454381  667263 buffer_comparator.cc:156] Difference at 232: 1451.91, expected 1192.73</span></span>
<span class="line"><span>E0000 00:00:1741386014.454384  667263 buffer_comparator.cc:156] Difference at 233: 1262.29, expected 1803.14</span></span>
<span class="line"><span>E0000 00:00:1741386014.454387  667263 buffer_comparator.cc:156] Difference at 234: 889.008, expected 1571.89</span></span>
<span class="line"><span>E0000 00:00:1741386014.454391  667263 buffer_comparator.cc:156] Difference at 235: 1418.52, expected 1102.22</span></span>
<span class="line"><span>2025-03-07 22:20:14.454398: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.457668  667263 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741386014.457688  667263 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741386014.457691  667263 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741386014.457695  667263 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741386014.457698  667263 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741386014.457701  667263 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741386014.457704  667263 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.457707  667263 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.457710  667263 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741386014.457713  667263 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-07 22:20:14.457720: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.460874  667263 buffer_comparator.cc:156] Difference at 7: 1058.92, expected 951.955</span></span>
<span class="line"><span>E0000 00:00:1741386014.460890  667263 buffer_comparator.cc:156] Difference at 11: 1263.92, expected 1121.95</span></span>
<span class="line"><span>E0000 00:00:1741386014.460895  667263 buffer_comparator.cc:156] Difference at 179: 1223.75, expected 1098.81</span></span>
<span class="line"><span>E0000 00:00:1741386014.460898  667263 buffer_comparator.cc:156] Difference at 266: 1047.35, expected 934.417</span></span>
<span class="line"><span>E0000 00:00:1741386014.460902  667263 buffer_comparator.cc:156] Difference at 270: 1246.8, expected 1101.55</span></span>
<span class="line"><span>E0000 00:00:1741386014.460906  667263 buffer_comparator.cc:156] Difference at 417: 1222.47, expected 1095.88</span></span>
<span class="line"><span>E0000 00:00:1741386014.460909  667263 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741386014.460912  667263 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741386014.460915  667263 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741386014.460918  667263 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>2025-03-07 22:20:14.460924: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.464125  667263 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741386014.464149  667263 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741386014.464152  667263 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741386014.464156  667263 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741386014.464159  667263 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741386014.464162  667263 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741386014.464165  667263 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.464168  667263 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.464171  667263 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741386014.464174  667263 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-07 22:20:14.464180: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.467398  667263 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741386014.467417  667263 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741386014.467420  667263 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741386014.467423  667263 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741386014.467427  667263 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741386014.467430  667263 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741386014.467433  667263 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.467436  667263 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.467439  667263 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741386014.467442  667263 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-07 22:20:14.467448: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.470597  667263 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741386014.470614  667263 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741386014.470617  667263 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741386014.470621  667263 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741386014.470624  667263 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741386014.470627  667263 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741386014.470630  667263 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.470633  667263 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.470636  667263 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741386014.470639  667263 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-07 22:20:14.470644: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.473852  667263 buffer_comparator.cc:156] Difference at 449: 1453.3, expected 1198.64</span></span>
<span class="line"><span>E0000 00:00:1741386014.473873  667263 buffer_comparator.cc:156] Difference at 450: 1279.09, expected 1813.5</span></span>
<span class="line"><span>E0000 00:00:1741386014.473876  667263 buffer_comparator.cc:156] Difference at 451: 886.266, expected 1575.24</span></span>
<span class="line"><span>E0000 00:00:1741386014.473880  667263 buffer_comparator.cc:156] Difference at 452: 1425.28, expected 1104.71</span></span>
<span class="line"><span>E0000 00:00:1741386014.473883  667263 buffer_comparator.cc:156] Difference at 453: 1023.45, expected 1764.88</span></span>
<span class="line"><span>E0000 00:00:1741386014.473886  667263 buffer_comparator.cc:156] Difference at 454: 741.257, expected 1269.7</span></span>
<span class="line"><span>E0000 00:00:1741386014.473889  667263 buffer_comparator.cc:156] Difference at 456: 1427.19, expected 1213.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.473892  667263 buffer_comparator.cc:156] Difference at 457: 1244.7, expected 1821.28</span></span>
<span class="line"><span>E0000 00:00:1741386014.473895  667263 buffer_comparator.cc:156] Difference at 458: 874.06, expected 1595.74</span></span>
<span class="line"><span>E0000 00:00:1741386014.473898  667263 buffer_comparator.cc:156] Difference at 459: 1393.56, expected 1114.22</span></span>
<span class="line"><span>2025-03-07 22:20:14.473905: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.477230  667263 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741386014.477246  667263 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741386014.477249  667263 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741386014.477255  667263 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741386014.477258  667263 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741386014.477261  667263 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.477264  667263 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741386014.477267  667263 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741386014.477270  667263 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741386014.477273  667263 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-07 22:20:14.477279: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.480550  667263 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741386014.480567  667263 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741386014.480570  667263 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741386014.480574  667263 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741386014.480577  667263 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741386014.480580  667263 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.480583  667263 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741386014.480586  667263 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741386014.480589  667263 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741386014.480592  667263 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-07 22:20:14.480597: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.483811  667263 buffer_comparator.cc:156] Difference at 897: 1448.31, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741386014.483832  667263 buffer_comparator.cc:156] Difference at 898: 1259.31, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741386014.483835  667263 buffer_comparator.cc:156] Difference at 899: 890.005, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741386014.483838  667263 buffer_comparator.cc:156] Difference at 900: 1414.65, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741386014.483841  667263 buffer_comparator.cc:156] Difference at 901: 1020.46, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741386014.483844  667263 buffer_comparator.cc:156] Difference at 902: 745.849, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.483848  667263 buffer_comparator.cc:156] Difference at 904: 1440.8, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741386014.483851  667263 buffer_comparator.cc:156] Difference at 905: 1259.53, expected 1817.42</span></span>
<span class="line"><span>E0000 00:00:1741386014.483854  667263 buffer_comparator.cc:156] Difference at 906: 875.773, expected 1572.98</span></span>
<span class="line"><span>E0000 00:00:1741386014.483857  667263 buffer_comparator.cc:156] Difference at 907: 1408.57, expected 1117.68</span></span>
<span class="line"><span>2025-03-07 22:20:14.483864: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.487096  667263 buffer_comparator.cc:156] Difference at 896: 485.098, expected 958.133</span></span>
<span class="line"><span>E0000 00:00:1741386014.487115  667263 buffer_comparator.cc:156] Difference at 897: 732.587, expected 1218.67</span></span>
<span class="line"><span>E0000 00:00:1741386014.487118  667263 buffer_comparator.cc:156] Difference at 898: 635.29, expected 1826.8</span></span>
<span class="line"><span>E0000 00:00:1741386014.487122  667263 buffer_comparator.cc:156] Difference at 899: 446.948, expected 1593.44</span></span>
<span class="line"><span>E0000 00:00:1741386014.487125  667263 buffer_comparator.cc:156] Difference at 900: 712.745, expected 1119.04</span></span>
<span class="line"><span>E0000 00:00:1741386014.487130  667263 buffer_comparator.cc:156] Difference at 901: 516.07, expected 1796.72</span></span>
<span class="line"><span>E0000 00:00:1741386014.487133  667263 buffer_comparator.cc:156] Difference at 902: 373.095, expected 1279.87</span></span>
<span class="line"><span>E0000 00:00:1741386014.487136  667263 buffer_comparator.cc:156] Difference at 903: 483.905, expected 941.483</span></span>
<span class="line"><span>E0000 00:00:1741386014.487139  667263 buffer_comparator.cc:156] Difference at 904: 721.412, expected 1202.98</span></span>
<span class="line"><span>E0000 00:00:1741386014.487142  667263 buffer_comparator.cc:156] Difference at 905: 633.571, expected 1817.42</span></span>
<span class="line"><span>2025-03-07 22:20:14.487148: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.490590  667263 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741386014.490613  667263 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741386014.490616  667263 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741386014.490620  667263 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.490623  667263 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.490626  667263 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741386014.490629  667263 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741386014.490632  667263 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741386014.490636  667263 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.490639  667263 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-07 22:20:14.490645: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.494109  667263 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741386014.494135  667263 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741386014.494138  667263 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741386014.494141  667263 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.494144  667263 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.494147  667263 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741386014.494150  667263 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741386014.494153  667263 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741386014.494157  667263 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.494160  667263 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-07 22:20:14.494169: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386014.497608  667263 buffer_comparator.cc:156] Difference at 1792: 481.623, expected 926.783</span></span>
<span class="line"><span>E0000 00:00:1741386014.497626  667263 buffer_comparator.cc:156] Difference at 1793: 723.158, expected 1190.76</span></span>
<span class="line"><span>E0000 00:00:1741386014.497629  667263 buffer_comparator.cc:156] Difference at 1794: 627.801, expected 1807.71</span></span>
<span class="line"><span>E0000 00:00:1741386014.497632  667263 buffer_comparator.cc:156] Difference at 1795: 438.191, expected 1565.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.497636  667263 buffer_comparator.cc:156] Difference at 1796: 703.772, expected 1101.05</span></span>
<span class="line"><span>E0000 00:00:1741386014.497639  667263 buffer_comparator.cc:156] Difference at 1797: 500.743, expected 1756.22</span></span>
<span class="line"><span>E0000 00:00:1741386014.497642  667263 buffer_comparator.cc:156] Difference at 1798: 384.438, expected 1272.35</span></span>
<span class="line"><span>E0000 00:00:1741386014.497647  667263 buffer_comparator.cc:156] Difference at 1799: 502.463, expected 944.47</span></span>
<span class="line"><span>E0000 00:00:1741386014.497650  667263 buffer_comparator.cc:156] Difference at 1800: 743.701, expected 1200.59</span></span>
<span class="line"><span>E0000 00:00:1741386014.497653  667263 buffer_comparator.cc:156] Difference at 1801: 656.627, expected 1808.37</span></span>
<span class="line"><span>2025-03-07 22:20:14.497659: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>2025-03-07 22:20:16.325244: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28_0&#39;, 36 bytes spill stores, 36 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:20:16.388630: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 4 bytes spill stores, 4 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>2025-03-07 22:20:16.473755: I external/xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function &#39;gemm_fusion_dot_28&#39;, 156 bytes spill stores, 156 bytes spill loads</span></span>
<span class="line"><span></span></span>
<span class="line"><span>E0000 00:00:1741386016.479567  667263 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741386016.479631  667263 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741386016.479640  667263 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741386016.479647  667263 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.479654  667263 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741386016.479660  667263 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741386016.479667  667263 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741386016.479674  667263 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741386016.479680  667263 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.479687  667263 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-07 22:20:16.479702: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.482264  667263 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741386016.482300  667263 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741386016.482308  667263 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741386016.482315  667263 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.482321  667263 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741386016.482328  667263 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741386016.482335  667263 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741386016.482341  667263 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741386016.482348  667263 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.482354  667263 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-07 22:20:16.482365: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.484912  667263 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741386016.484940  667263 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741386016.484948  667263 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741386016.484955  667263 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.484963  667263 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741386016.484970  667263 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741386016.484977  667263 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741386016.484983  667263 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741386016.484990  667263 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.484996  667263 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-07 22:20:16.485007: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.487538  667263 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741386016.487555  667263 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741386016.487558  667263 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741386016.487561  667263 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.487564  667263 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741386016.487567  667263 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741386016.487570  667263 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741386016.487573  667263 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741386016.487576  667263 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.487579  667263 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-07 22:20:16.487584: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.489846  667263 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741386016.489867  667263 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741386016.489870  667263 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741386016.489873  667263 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.489876  667263 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741386016.489879  667263 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741386016.489882  667263 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741386016.489885  667263 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741386016.489888  667263 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.489891  667263 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-07 22:20:16.489896: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.492177  667263 buffer_comparator.cc:156] Difference at 112: 0, expected 769.985</span></span>
<span class="line"><span>E0000 00:00:1741386016.492195  667263 buffer_comparator.cc:156] Difference at 113: 0, expected 1100</span></span>
<span class="line"><span>E0000 00:00:1741386016.492198  667263 buffer_comparator.cc:156] Difference at 114: 0, expected 1061.37</span></span>
<span class="line"><span>E0000 00:00:1741386016.492201  667263 buffer_comparator.cc:156] Difference at 115: 0, expected 1558.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.492204  667263 buffer_comparator.cc:156] Difference at 116: 0, expected 1573.39</span></span>
<span class="line"><span>E0000 00:00:1741386016.492207  667263 buffer_comparator.cc:156] Difference at 117: 0, expected 1297.47</span></span>
<span class="line"><span>E0000 00:00:1741386016.492210  667263 buffer_comparator.cc:156] Difference at 118: 0, expected 880.235</span></span>
<span class="line"><span>E0000 00:00:1741386016.492213  667263 buffer_comparator.cc:156] Difference at 119: 0, expected 764.244</span></span>
<span class="line"><span>E0000 00:00:1741386016.492218  667263 buffer_comparator.cc:156] Difference at 120: 0, expected 1089.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.492221  667263 buffer_comparator.cc:156] Difference at 121: 0, expected 1044.63</span></span>
<span class="line"><span>2025-03-07 22:20:16.492225: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.494461  667263 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741386016.494478  667263 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741386016.494481  667263 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741386016.494484  667263 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741386016.494487  667263 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741386016.494490  667263 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741386016.494493  667263 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741386016.494496  667263 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741386016.494499  667263 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741386016.494502  667263 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-07 22:20:16.494507: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.496743  667263 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741386016.496763  667263 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741386016.496766  667263 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741386016.496769  667263 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741386016.496772  667263 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741386016.496775  667263 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741386016.496778  667263 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741386016.496781  667263 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741386016.496784  667263 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741386016.496787  667263 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-07 22:20:16.496792: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.499029  667263 buffer_comparator.cc:156] Difference at 224: 0, expected 745.838</span></span>
<span class="line"><span>E0000 00:00:1741386016.499049  667263 buffer_comparator.cc:156] Difference at 225: 0, expected 1079.1</span></span>
<span class="line"><span>E0000 00:00:1741386016.499052  667263 buffer_comparator.cc:156] Difference at 226: 0, expected 1034.99</span></span>
<span class="line"><span>E0000 00:00:1741386016.499055  667263 buffer_comparator.cc:156] Difference at 227: 0, expected 1538.8</span></span>
<span class="line"><span>E0000 00:00:1741386016.499058  667263 buffer_comparator.cc:156] Difference at 228: 0, expected 1554.44</span></span>
<span class="line"><span>E0000 00:00:1741386016.499061  667263 buffer_comparator.cc:156] Difference at 229: 0, expected 1264.82</span></span>
<span class="line"><span>E0000 00:00:1741386016.499064  667263 buffer_comparator.cc:156] Difference at 230: 0, expected 853.966</span></span>
<span class="line"><span>E0000 00:00:1741386016.499066  667263 buffer_comparator.cc:156] Difference at 231: 0, expected 756.177</span></span>
<span class="line"><span>E0000 00:00:1741386016.499069  667263 buffer_comparator.cc:156] Difference at 232: 0, expected 1076.91</span></span>
<span class="line"><span>E0000 00:00:1741386016.499072  667263 buffer_comparator.cc:156] Difference at 233: 0, expected 1029.02</span></span>
<span class="line"><span>2025-03-07 22:20:16.499077: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.501332  667263 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741386016.501347  667263 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741386016.501350  667263 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741386016.501353  667263 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741386016.501356  667263 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741386016.501359  667263 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741386016.501362  667263 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741386016.501365  667263 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741386016.501367  667263 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741386016.501370  667263 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-07 22:20:16.501375: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.503597  667263 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741386016.503617  667263 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741386016.503620  667263 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741386016.503623  667263 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741386016.503626  667263 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741386016.503629  667263 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741386016.503632  667263 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741386016.503634  667263 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741386016.503637  667263 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741386016.503640  667263 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-07 22:20:16.503645: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.505942  667263 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741386016.505958  667263 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741386016.505961  667263 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741386016.505964  667263 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741386016.505968  667263 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741386016.505970  667263 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741386016.505973  667263 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741386016.505976  667263 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741386016.505979  667263 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741386016.505982  667263 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-07 22:20:16.505987: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.508228  667263 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741386016.508244  667263 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741386016.508247  667263 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741386016.508252  667263 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741386016.508255  667263 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741386016.508258  667263 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741386016.508261  667263 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741386016.508263  667263 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741386016.508266  667263 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741386016.508269  667263 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-07 22:20:16.508274: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.510500  667263 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741386016.510519  667263 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741386016.510523  667263 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741386016.510526  667263 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741386016.510529  667263 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741386016.510532  667263 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741386016.510534  667263 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741386016.510537  667263 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741386016.510540  667263 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741386016.510543  667263 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-07 22:20:16.510548: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.512766  667263 buffer_comparator.cc:156] Difference at 448: 0, expected 770.258</span></span>
<span class="line"><span>E0000 00:00:1741386016.512782  667263 buffer_comparator.cc:156] Difference at 449: 0, expected 1098.93</span></span>
<span class="line"><span>E0000 00:00:1741386016.512785  667263 buffer_comparator.cc:156] Difference at 450: 0, expected 1056.29</span></span>
<span class="line"><span>E0000 00:00:1741386016.512788  667263 buffer_comparator.cc:156] Difference at 451: 0, expected 1560.21</span></span>
<span class="line"><span>E0000 00:00:1741386016.512791  667263 buffer_comparator.cc:156] Difference at 452: 0, expected 1585.41</span></span>
<span class="line"><span>E0000 00:00:1741386016.512794  667263 buffer_comparator.cc:156] Difference at 453: 0, expected 1307.15</span></span>
<span class="line"><span>E0000 00:00:1741386016.512797  667263 buffer_comparator.cc:156] Difference at 454: 0, expected 881.296</span></span>
<span class="line"><span>E0000 00:00:1741386016.512800  667263 buffer_comparator.cc:156] Difference at 455: 0, expected 760.638</span></span>
<span class="line"><span>E0000 00:00:1741386016.512803  667263 buffer_comparator.cc:156] Difference at 456: 0, expected 1092.67</span></span>
<span class="line"><span>E0000 00:00:1741386016.512806  667263 buffer_comparator.cc:156] Difference at 457: 0, expected 1051.03</span></span>
<span class="line"><span>2025-03-07 22:20:16.512811: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.515303  667263 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741386016.515320  667263 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.515323  667263 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.515326  667263 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741386016.515329  667263 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741386016.515332  667263 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741386016.515335  667263 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741386016.515339  667263 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741386016.515342  667263 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741386016.515345  667263 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-07 22:20:16.515350: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.517636  667263 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741386016.517652  667263 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.517655  667263 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.517658  667263 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741386016.517661  667263 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741386016.517664  667263 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741386016.517667  667263 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741386016.517670  667263 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741386016.517673  667263 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741386016.517675  667263 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-07 22:20:16.517680: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.520132  667263 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741386016.520150  667263 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.520154  667263 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.520157  667263 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741386016.520159  667263 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741386016.520162  667263 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741386016.520165  667263 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741386016.520168  667263 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741386016.520171  667263 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741386016.520174  667263 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-07 22:20:16.520179: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.522423  667263 buffer_comparator.cc:156] Difference at 896: 0, expected 767.869</span></span>
<span class="line"><span>E0000 00:00:1741386016.522439  667263 buffer_comparator.cc:156] Difference at 897: 0, expected 1090.2</span></span>
<span class="line"><span>E0000 00:00:1741386016.522442  667263 buffer_comparator.cc:156] Difference at 898: 0, expected 1050.23</span></span>
<span class="line"><span>E0000 00:00:1741386016.522445  667263 buffer_comparator.cc:156] Difference at 899: 0, expected 1561.6</span></span>
<span class="line"><span>E0000 00:00:1741386016.522448  667263 buffer_comparator.cc:156] Difference at 900: 0, expected 1574.44</span></span>
<span class="line"><span>E0000 00:00:1741386016.522451  667263 buffer_comparator.cc:156] Difference at 901: 0, expected 1303.84</span></span>
<span class="line"><span>E0000 00:00:1741386016.522454  667263 buffer_comparator.cc:156] Difference at 902: 0, expected 881.498</span></span>
<span class="line"><span>E0000 00:00:1741386016.522457  667263 buffer_comparator.cc:156] Difference at 903: 0, expected 755.455</span></span>
<span class="line"><span>E0000 00:00:1741386016.522460  667263 buffer_comparator.cc:156] Difference at 904: 0, expected 1073.52</span></span>
<span class="line"><span>E0000 00:00:1741386016.522462  667263 buffer_comparator.cc:156] Difference at 905: 0, expected 1034.81</span></span>
<span class="line"><span>2025-03-07 22:20:16.522467: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.524835  667263 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741386016.524851  667263 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741386016.524854  667263 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741386016.524857  667263 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741386016.524860  667263 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741386016.524862  667263 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741386016.524865  667263 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741386016.524868  667263 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741386016.524871  667263 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741386016.524874  667263 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-07 22:20:16.524879: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.527395  667263 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741386016.527412  667263 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741386016.527416  667263 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741386016.527419  667263 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741386016.527422  667263 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741386016.527424  667263 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741386016.527427  667263 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741386016.527430  667263 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741386016.527433  667263 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741386016.527436  667263 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-07 22:20:16.527441: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>E0000 00:00:1741386016.529788  667263 buffer_comparator.cc:156] Difference at 1792: 0, expected 748.592</span></span>
<span class="line"><span>E0000 00:00:1741386016.529804  667263 buffer_comparator.cc:156] Difference at 1793: 0, expected 1073.49</span></span>
<span class="line"><span>E0000 00:00:1741386016.529807  667263 buffer_comparator.cc:156] Difference at 1794: 0, expected 1027.26</span></span>
<span class="line"><span>E0000 00:00:1741386016.529810  667263 buffer_comparator.cc:156] Difference at 1795: 0, expected 1535.73</span></span>
<span class="line"><span>E0000 00:00:1741386016.529813  667263 buffer_comparator.cc:156] Difference at 1796: 0, expected 1559.13</span></span>
<span class="line"><span>E0000 00:00:1741386016.529816  667263 buffer_comparator.cc:156] Difference at 1797: 0, expected 1277.09</span></span>
<span class="line"><span>E0000 00:00:1741386016.529819  667263 buffer_comparator.cc:156] Difference at 1798: 0, expected 859.43</span></span>
<span class="line"><span>E0000 00:00:1741386016.529822  667263 buffer_comparator.cc:156] Difference at 1799: 0, expected 752.412</span></span>
<span class="line"><span>E0000 00:00:1741386016.529825  667263 buffer_comparator.cc:156] Difference at 1800: 0, expected 1077.59</span></span>
<span class="line"><span>E0000 00:00:1741386016.529828  667263 buffer_comparator.cc:156] Difference at 1801: 0, expected 1037.98</span></span>
<span class="line"><span>2025-03-07 22:20:16.529833: E external/xla/xla/service/gpu/autotuning/gemm_fusion_autotuner.cc:1138] Results do not match the reference. This is likely a bug/unexpected loss of precision.</span></span>
<span class="line"><span>Test Loss: 1.505029	Test Acc: 67.8000%</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MLDataDevices)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.3</span></span>
<span class="line"><span>Commit d63adeda50d (2025-01-21 19:42 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,21)]))}const u=s(c,[["render",i]]);export{d as __pageData,u as default};
