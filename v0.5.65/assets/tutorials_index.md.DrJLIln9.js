import{d as c,o,c as r,j as e,k as g,g as f,t as u,_ as m,F as _,E as b,b as v,M as w,I as s,a}from"./chunks/framework.C_hFR9fe.js";const y={class:"img-box"},x=["href"],N=["src"],L={class:"transparent-box1"},T={class:"caption"},k={class:"transparent-box2"},D={class:"subcaption"},P={class:"opacity-low"},I=c({__name:"GalleryImage",props:{href:{},src:{},caption:{},desc:{}},setup(d){return(t,n)=>(o(),r("div",y,[e("a",{href:t.href},[e("img",{src:g(f)(t.src),height:"150px",alt:""},null,8,N),e("div",L,[e("div",T,[e("h2",null,u(t.caption),1)])]),e("div",k,[e("div",D,[e("p",P,u(t.desc),1)])])],8,x)]))}}),C=m(I,[["__scopeId","data-v-06a0366f"]]),E={class:"gallery-image"},M=c({__name:"Gallery",props:{images:{}},setup(d){return(t,n)=>(o(),r("div",E,[(o(!0),r(_,null,b(t.images,l=>(o(),v(C,w({ref_for:!0},l),null,16))),256))]))}}),i=m(M,[["__scopeId","data-v-578d61bc"]]),S=e("h1",{id:"tutorials",tabindex:"-1"},[a("Tutorials "),e("a",{class:"header-anchor",href:"#tutorials","aria-label":'Permalink to "Tutorials"'},"​")],-1),j=e("h2",{id:"beginner-tutorials",tabindex:"-1"},[a("Beginner Tutorials "),e("a",{class:"header-anchor",href:"#beginner-tutorials","aria-label":'Permalink to "Beginner Tutorials"'},"​")],-1),B=e("h2",{id:"intermediate-tutorials",tabindex:"-1"},[a("Intermediate Tutorials "),e("a",{class:"header-anchor",href:"#intermediate-tutorials","aria-label":'Permalink to "Intermediate Tutorials"'},"​")],-1),F=e("h2",{id:"advanced-tutorials",tabindex:"-1"},[a("Advanced Tutorials "),e("a",{class:"header-anchor",href:"#advanced-tutorials","aria-label":'Permalink to "Advanced Tutorials"'},"​")],-1),G=e("h2",{id:"larger-models",tabindex:"-1"},[a("Larger Models "),e("a",{class:"header-anchor",href:"#larger-models","aria-label":'Permalink to "Larger Models"'},"​")],-1),O=e("div",{class:"warning custom-block"},[e("p",{class:"custom-block-title"},"WARNING"),e("p",null,"These models are part of the Lux examples, however, these are larger model that cannot be run on CI and aren't frequently tested. If you find a bug in one of these models, please open an issue or PR to fix it.")],-1),$=e("h2",{id:"selected-3rd-party-tutorials",tabindex:"-1"},[a("Selected 3rd Party Tutorials "),e("a",{class:"header-anchor",href:"#selected-3rd-party-tutorials","aria-label":'Permalink to "Selected 3rd Party Tutorials"'},"​")],-1),R=e("div",{class:"warning custom-block"},[e("p",{class:"custom-block-title"},"WARNING"),e("p",null,[a("These tutorials are developed by the community and may not be up-to-date with the latest version of "),e("code",null,"Lux.jl"),a(". Please refer to the official documentation for the most up-to-date information.")]),e("p",null,[a("Please open an issue (ideally both at "),e("code",null,"Lux.jl"),a(" and at the downstream linked package) if any of them are non-functional and we will try to get them updated.")])],-1),U=e("div",{class:"tip custom-block"},[e("p",{class:"custom-block-title"},"TIP"),e("p",null,[a("If you found an amazing tutorial showcasing "),e("code",null,"Lux.jl"),a(" online, or wrote one yourself, please open an issue or PR to add it to the list!")])],-1),W=JSON.parse('{"title":"Tutorials","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/index.md","filePath":"tutorials/index.md","lastUpdated":null}'),A={name:"tutorials/index.md"},z=c({...A,setup(d){const t=[{href:"beginner/1_Basics",src:"https://picsum.photos/350/250?image=444",caption:"Julia & Lux for the Uninitiated",desc:"How to get started with Julia and Lux for those who have never used Julia before."},{href:"beginner/2_PolynomialFitting",src:"../mlp.webp",caption:"Fitting a Polynomial using MLP",desc:"Learn the Basics of Lux by fitting a Multi-Layer Perceptron to a Polynomial."},{href:"beginner/3_SimpleRNN",src:"../lstm-illustrative.webp",caption:"Training a Simple LSTM",desc:"Learn how to define custom layers and train an RNN on time-series data."},{href:"beginner/4_SimpleChains",src:"../blas_optimizations.jpg",caption:"Use SimpleChains.jl as a Backend",desc:"Learn how to train small neural networks really fast on CPU."}],n=[{href:"intermediate/1_NeuralODE",src:"../mnist.jpg",caption:"MNIST Classification using Neural ODE",desc:"Train a Neural Ordinary Differential Equations to classify MNIST Images."},{href:"intermediate/2_BayesianNN",src:"https://github.com/TuringLang.png",caption:"Bayesian Neural Networks",desc:"Figure out how to use Probabilistic Programming Frameworks like Turing with Lux."},{href:"intermediate/3_HyperNet",src:"../hypernet.jpg",caption:"Training a HyperNetwork",desc:"Train a hypernetwork to work on multiple datasets by predicting neural network parameters."}],l=[{href:"advanced/1_GravitationalWaveForm",src:"../gravitational_waveform.png",caption:"Neural ODE to Model Gravitational Waveforms",desc:"Training a Neural ODE to fit simulated data of gravitational waveforms."},{href:"advanced/2_SymbolicOptimalControl",src:"../symbolic_optimal_control.png",caption:"Optimal Control with Symbolic UDE",desc:"Train a UDE and replace a part of it with Symbolic Regression."}],p=[{href:"https://github.com/LuxDL/Lux.jl/tree/main/examples/ImageNet",src:"https://production-media.paperswithcode.com/datasets/ImageNet-0000000008-f2e87edd_Y0fT5zg.jpg",caption:"ImageNet Classification",desc:"Train Large Image Classifiers using Lux (on Distributed GPUs)."},{href:"https://github.com/LuxDL/Lux.jl/tree/main/examples/DDIM",src:"https://raw.githubusercontent.com/LuxDL/Lux.jl/main/examples/DDIM/assets/flowers_generated.png",caption:"Denoising Diffusion Implicit Model (DDIM)",desc:"Train a Diffusion Model to generate images from Gaussian noises."},{href:"https://github.com/LuxDL/Lux.jl/tree/main/examples/ConvMixer",src:"https://datasets.activeloop.ai/wp-content/uploads/2022/09/CIFAR-10-dataset-Activeloop-Platform-visualization-image-1.webp",caption:"ConvMixer on CIFAR-10",desc:"Train ConvMixer on CIFAR-10 to 90% accuracy within 10 minutes."}],h=[{href:"https://docs.sciml.ai/Overview/stable/showcase/pinngpu/",src:"../pinn.gif",caption:"GPU-Accelerated Physics-Informed Neural Networks",desc:"Use Machine Learning (PINNs) to solve the Heat Equation PDE on a GPU."},{href:"https://docs.sciml.ai/DiffEqFlux/stable/examples/neural_ode_weather_forecast/",src:"../weather-neural-ode.gif",caption:"Weather Forecasting with Neural ODEs",desc:"Train a neural ODEs to a multidimensional weather dataset and use it for weather forecasting."},{href:"https://docs.sciml.ai/SciMLSensitivity/stable/examples/sde/SDE_control/",src:"../neural-sde.png",caption:"Controlling Stochastic Differential Equations",desc:"Control the time evolution of a continuously monitored qubit described by an SDE with multiplicative scalar noise."},{href:"https://github.com/Dale-Black/ComputerVisionTutorials.jl/",src:"https://raw.githubusercontent.com/Dale-Black/ComputerVisionTutorials.jl/main/assets/image-seg-green.jpeg",caption:"Medical Image Segmentation",desc:"Explore various aspects of deep learning for medical imaging and a comprehensive overview of Julia packages."},{href:"https://github.com/agdestein/NeuralClosureTutorials",src:"https://raw.githubusercontent.com/agdestein/NeuralClosureTutorials/main/assets/navier_stokes.gif",caption:"Neural PDE closures",desc:"Learn an unknown term in a PDE using convolutional neural networks and Fourier neural operators."}];return(q,J)=>(o(),r("div",null,[S,j,s(i,{images:t}),B,s(i,{images:n}),F,s(i,{images:l}),G,O,s(i,{images:p}),$,R,s(i,{images:h}),U]))}});export{W as __pageData,z as default};
