import{_ as l,c as o,j as t,a as e,a3 as a,o as s}from"./chunks/framework.B5P1ePk0.js";const J=JSON.parse('{"title":"Automatic Differentiation","description":"","frontmatter":{},"headers":[],"relativePath":"api/Lux/autodiff.md","filePath":"api/Lux/autodiff.md","lastUpdated":null}'),i={name:"api/Lux/autodiff.md"},r=a('<h1 id="Automatic-Differentiation" tabindex="-1">Automatic Differentiation <a class="header-anchor" href="#Automatic-Differentiation" aria-label="Permalink to &quot;Automatic Differentiation {#Automatic-Differentiation}&quot;">​</a></h1><p>Lux is not an AD package, but it composes well with most of the AD packages available in the Julia ecosystem. This document lists the current level of support for various AD packages in Lux. Additionally, we provide some convenience functions for working with AD.</p><h2 id="Overview" tabindex="-1">Overview <a class="header-anchor" href="#Overview" aria-label="Permalink to &quot;Overview {#Overview}&quot;">​</a></h2><table><thead><tr><th style="text-align:left;">AD Package</th><th style="text-align:left;">CPU</th><th style="text-align:left;">GPU</th><th style="text-align:left;">Nested 2nd Order AD</th><th style="text-align:left;">Support Class</th></tr></thead><tbody><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ChainRules.jl" target="_blank" rel="noreferrer"><code>ChainRules.jl</code></a><sup class="footnote-ref"><a href="#fn1" id="fnref1">[1]</a></sup></td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/FluxML/Zygote.jl" target="_blank" rel="noreferrer"><code>Zygote.jl</code></a></td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ForwardDiff.jl" target="_blank" rel="noreferrer"><code>ForwardDiff.jl</code></a></td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">Tier I</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/ReverseDiff.jl" target="_blank" rel="noreferrer"><code>ReverseDiff.jl</code></a></td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier II</td></tr><tr><td style="text-align:left;"><a href="https://github.com/FluxML/Tracker.jl" target="_blank" rel="noreferrer"><code>Tracker.jl</code></a></td><td style="text-align:left;">✔️</td><td style="text-align:left;">✔️</td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier II</td></tr><tr><td style="text-align:left;"><a href="https://github.com/EnzymeAD/Enzyme.jl" target="_blank" rel="noreferrer"><code>Enzyme.jl</code></a></td><td style="text-align:left;">✔️</td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2">[2]</a></sup></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:1">[2:1]</a></sup></td><td style="text-align:left;">Tier II</td></tr><tr><td style="text-align:left;"><a href="https://github.com/withbayes/Tapir.jl" target="_blank" rel="noreferrer"><code>Tapir.jl</code></a></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:2">[2:2]</a></sup></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:3">[2:3]</a></sup></td><td style="text-align:left;">❌</td><td style="text-align:left;">Tier IV</td></tr><tr><td style="text-align:left;"><a href="https://github.com/JuliaDiff/Diffractor.jl" target="_blank" rel="noreferrer"><code>Diffractor.jl</code></a></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:4">[2:4]</a></sup></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:5">[2:5]</a></sup></td><td style="text-align:left;">❓<sup class="footnote-ref"><a href="#fn2" id="fnref2:6">[2:6]</a></sup></td><td style="text-align:left;">Tier IV</td></tr></tbody></table><h3 id="Support-Class" tabindex="-1">Support Class <a class="header-anchor" href="#Support-Class" aria-label="Permalink to &quot;Support Class {#Support-Class}&quot;">​</a></h3><ol><li><p><strong>Tier I</strong>: These packages are fully supported and have been tested extensively. Often have special rules to enhance performance. Issues for these backends take the highest priority.</p></li><li><p><strong>Tier II</strong>: These packages are supported and extensively tested but often don&#39;t have the best performance. Issues against these backends are less critical, but we fix them when possible. (Some specific edge cases, especially with AMDGPU, are known to fail here)</p></li><li><p><strong>Tier III</strong>: These packages are somewhat tested but expect rough edges. Help us add tests for these backends to get them to Tier II status.</p></li><li><p><strong>Tier IV</strong>: We don&#39;t know if these packages currently work with Lux. We&#39;d love to add tests for these backends, but currently these are not our priority.</p></li></ol><h2 id="Index" tabindex="-1">Index <a class="header-anchor" href="#Index" aria-label="Permalink to &quot;Index {#Index}&quot;">​</a></h2><ul><li><a href="#Lux.batched_jacobian"><code>Lux.batched_jacobian</code></a></li><li><a href="#Lux.jacobian_vector_product"><code>Lux.jacobian_vector_product</code></a></li><li><a href="#Lux.vector_jacobian_product"><code>Lux.vector_jacobian_product</code></a></li></ul><h2 id="JVP-and-JVP-Wrappers" tabindex="-1">JVP &amp; JVP Wrappers <a class="header-anchor" href="#JVP-and-JVP-Wrappers" aria-label="Permalink to &quot;JVP &amp;amp; JVP Wrappers {#JVP-and-JVP-Wrappers}&quot;">​</a></h2>',9),d={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},n=t("a",{id:"Lux.jacobian_vector_product",href:"#Lux.jacobian_vector_product"},"#",-1),c=t("b",null,[t("u",null,"Lux.jacobian_vector_product")],-1),h=t("i",null,"Function",-1),p=a('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">jacobian_vector_product</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractADType</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x, u)</span></span></code></pre></div>',1),f={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},T={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-1.469ex"},xmlns:"http://www.w3.org/2000/svg",width:"6.812ex",height:"4.07ex",role:"img",focusable:"false",viewBox:"0 -1149.5 3010.7 1799","aria-hidden":"true"},u=a('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="mrow"><g data-mml-node="mo" transform="translate(0 -0.5)"><path data-c="28" d="M180 96T180 250T205 541T266 770T353 944T444 1069T527 1150H555Q561 1144 561 1141Q561 1137 545 1120T504 1072T447 995T386 878T330 721T288 513T272 251Q272 133 280 56Q293 -87 326 -209T399 -405T475 -531T536 -609T561 -640Q561 -643 555 -649H527Q483 -612 443 -568T353 -443T266 -270T205 -41Z" style="stroke-width:3;"></path></g><g data-mml-node="mfrac" transform="translate(597,0)"><g data-mml-node="mrow" transform="translate(227.8,485) scale(0.707)"><g data-mml-node="mi"><path data-c="1D715" d="M202 508Q179 508 169 520T158 547Q158 557 164 577T185 624T230 675T301 710L333 715H345Q378 715 384 714Q447 703 489 661T549 568T566 457Q566 362 519 240T402 53Q321 -22 223 -22Q123 -22 73 56Q42 102 42 148V159Q42 276 129 370T322 465Q383 465 414 434T455 367L458 378Q478 461 478 515Q478 603 437 639T344 676Q266 676 223 612Q264 606 264 572Q264 547 246 528T202 508ZM430 306Q430 372 401 400T333 428Q270 428 222 382Q197 354 183 323T150 221Q132 149 132 116Q132 21 232 21Q244 21 250 22Q327 35 374 112Q389 137 409 196T430 306Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(566,0)"><path data-c="1D453" d="M118 -162Q120 -162 124 -164T135 -167T147 -168Q160 -168 171 -155T187 -126Q197 -99 221 27T267 267T289 382V385H242Q195 385 192 387Q188 390 188 397L195 425Q197 430 203 430T250 431Q298 431 298 432Q298 434 307 482T319 540Q356 705 465 705Q502 703 526 683T550 630Q550 594 529 578T487 561Q443 561 443 603Q443 622 454 636T478 657L487 662Q471 668 457 668Q445 668 434 658T419 630Q412 601 403 552T387 469T380 433Q380 431 435 431Q480 431 487 430T498 424Q499 420 496 407T491 391Q489 386 482 386T428 385H372L349 263Q301 15 282 -47Q255 -132 212 -173Q175 -205 139 -205Q107 -205 81 -186T55 -132Q55 -95 76 -78T118 -61Q162 -61 162 -103Q162 -122 151 -136T127 -157L118 -162Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mrow" transform="translate(220,-345.6) scale(0.707)"><g data-mml-node="mi"><path data-c="1D715" d="M202 508Q179 508 169 520T158 547Q158 557 164 577T185 624T230 675T301 710L333 715H345Q378 715 384 714Q447 703 489 661T549 568T566 457Q566 362 519 240T402 53Q321 -22 223 -22Q123 -22 73 56Q42 102 42 148V159Q42 276 129 370T322 465Q383 465 414 434T455 367L458 378Q478 461 478 515Q478 603 437 639T344 676Q266 676 223 612Q264 606 264 572Q264 547 246 528T202 508ZM430 306Q430 372 401 400T333 428Q270 428 222 382Q197 354 183 323T150 221Q132 149 132 116Q132 21 232 21Q244 21 250 22Q327 35 374 112Q389 137 409 196T430 306Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(566,0)"><path data-c="1D465" d="M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z" style="stroke-width:3;"></path></g></g><rect width="1004.7" height="60" x="120" y="220"></rect></g><g data-mml-node="mo" transform="translate(1841.7,0) translate(0 -0.5)"><path data-c="29" d="M35 1138Q35 1150 51 1150H56H69Q113 1113 153 1069T243 944T330 771T391 541T416 250T391 -40T330 -270T243 -443T152 -568T69 -649H56Q43 -649 39 -647T35 -637Q65 -607 110 -548Q283 -316 316 56Q324 133 324 251Q324 368 316 445Q278 877 48 1123Q36 1137 35 1138Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mi" transform="translate(2438.7,0)"><path data-c="1D462" d="M21 287Q21 295 30 318T55 370T99 420T158 442Q204 442 227 417T250 358Q250 340 216 246T182 105Q182 62 196 45T238 27T291 44T328 78L339 95Q341 99 377 247Q407 367 413 387T427 416Q444 431 463 431Q480 431 488 421T496 402L420 84Q419 79 419 68Q419 43 426 35T447 26Q469 29 482 57T512 145Q514 153 532 153Q551 153 551 144Q550 139 549 130T540 98T523 55T498 17T462 -8Q454 -10 438 -10Q372 -10 347 46Q345 45 336 36T318 21T296 6T267 -6T233 -11Q189 -11 155 7Q103 38 103 113Q103 170 138 262T173 379Q173 380 173 381Q173 390 173 393T169 400T158 404H154Q131 404 112 385T82 344T65 302T57 280Q55 278 41 278H27Q21 284 21 287Z" style="stroke-width:3;"></path></g></g></g>',1),Q=[u],g=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"("),t("mfrac",null,[t("mrow",null,[t("mi",null,"∂"),t("mi",null,"f")]),t("mrow",null,[t("mi",null,"∂"),t("mi",null,"x")])]),t("mo",{"data-mjx-texclass":"CLOSE"},")")]),t("mi",null,"u")])],-1),m=a('<p><strong>Backends &amp; AD Packages</strong></p><table><thead><tr><th style="text-align:left;">Supported Backends</th><th style="text-align:left;">Packages Needed</th></tr></thead><tbody><tr><td style="text-align:left;"><code>AutoForwardDiff</code></td><td style="text-align:left;"><code>ForwardDiff.jl</code></td></tr></tbody></table><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>Gradient wrt <code>u</code> in the reverse pass is always dropped.</p></div><p><strong>Arguments</strong></p><ul><li><p><code>f</code>: The function to compute the jacobian of.</p></li><li><p><code>backend</code>: The backend to use for computing the JVP.</p></li><li><p><code>x</code>: The input to the function.</p></li><li><p><code>u</code>: An object of the same structure as <code>x</code>.</p></li></ul><p><strong>Returns</strong></p><ul><li><code>v</code>: The Jacobian Vector Product.</li></ul><p><a href="https://github.com/LuxDL/Lux.jl/blob/baa2092615116194ed3be513dc2f6b8126877de0/src/helpers/autodiff.jl#L41" target="_blank" rel="noreferrer">source</a></p>',8),b=t("br",null,null,-1),x={style:{"border-width":"1px","border-style":"solid","border-color":"black",padding:"1em","border-radius":"25px"}},_=t("a",{id:"Lux.vector_jacobian_product",href:"#Lux.vector_jacobian_product"},"#",-1),k=t("b",null,[t("u",null,"Lux.vector_jacobian_product")],-1),y=t("i",null,"Function",-1),w=a('<div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vector_jacobian_product</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractADType</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x, u)</span></span></code></pre></div>',1),v={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},A={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-1.469ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.126ex",height:"4.536ex",role:"img",focusable:"false",viewBox:"0 -1355.3 3591.5 2004.8","aria-hidden":"true"},D=a('<g stroke="currentColor" fill="currentColor" stroke-width="0" transform="scale(1,-1)"><g data-mml-node="math"><g data-mml-node="msup"><g data-mml-node="mrow"><g data-mml-node="mo" transform="translate(0 -0.5)"><path data-c="28" d="M180 96T180 250T205 541T266 770T353 944T444 1069T527 1150H555Q561 1144 561 1141Q561 1137 545 1120T504 1072T447 995T386 878T330 721T288 513T272 251Q272 133 280 56Q293 -87 326 -209T399 -405T475 -531T536 -609T561 -640Q561 -643 555 -649H527Q483 -612 443 -568T353 -443T266 -270T205 -41Z" style="stroke-width:3;"></path></g><g data-mml-node="mfrac" transform="translate(597,0)"><g data-mml-node="mrow" transform="translate(227.8,485) scale(0.707)"><g data-mml-node="mi"><path data-c="1D715" d="M202 508Q179 508 169 520T158 547Q158 557 164 577T185 624T230 675T301 710L333 715H345Q378 715 384 714Q447 703 489 661T549 568T566 457Q566 362 519 240T402 53Q321 -22 223 -22Q123 -22 73 56Q42 102 42 148V159Q42 276 129 370T322 465Q383 465 414 434T455 367L458 378Q478 461 478 515Q478 603 437 639T344 676Q266 676 223 612Q264 606 264 572Q264 547 246 528T202 508ZM430 306Q430 372 401 400T333 428Q270 428 222 382Q197 354 183 323T150 221Q132 149 132 116Q132 21 232 21Q244 21 250 22Q327 35 374 112Q389 137 409 196T430 306Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(566,0)"><path data-c="1D453" d="M118 -162Q120 -162 124 -164T135 -167T147 -168Q160 -168 171 -155T187 -126Q197 -99 221 27T267 267T289 382V385H242Q195 385 192 387Q188 390 188 397L195 425Q197 430 203 430T250 431Q298 431 298 432Q298 434 307 482T319 540Q356 705 465 705Q502 703 526 683T550 630Q550 594 529 578T487 561Q443 561 443 603Q443 622 454 636T478 657L487 662Q471 668 457 668Q445 668 434 658T419 630Q412 601 403 552T387 469T380 433Q380 431 435 431Q480 431 487 430T498 424Q499 420 496 407T491 391Q489 386 482 386T428 385H372L349 263Q301 15 282 -47Q255 -132 212 -173Q175 -205 139 -205Q107 -205 81 -186T55 -132Q55 -95 76 -78T118 -61Q162 -61 162 -103Q162 -122 151 -136T127 -157L118 -162Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mrow" transform="translate(220,-345.6) scale(0.707)"><g data-mml-node="mi"><path data-c="1D715" d="M202 508Q179 508 169 520T158 547Q158 557 164 577T185 624T230 675T301 710L333 715H345Q378 715 384 714Q447 703 489 661T549 568T566 457Q566 362 519 240T402 53Q321 -22 223 -22Q123 -22 73 56Q42 102 42 148V159Q42 276 129 370T322 465Q383 465 414 434T455 367L458 378Q478 461 478 515Q478 603 437 639T344 676Q266 676 223 612Q264 606 264 572Q264 547 246 528T202 508ZM430 306Q430 372 401 400T333 428Q270 428 222 382Q197 354 183 323T150 221Q132 149 132 116Q132 21 232 21Q244 21 250 22Q327 35 374 112Q389 137 409 196T430 306Z" style="stroke-width:3;"></path></g><g data-mml-node="mi" transform="translate(566,0)"><path data-c="1D465" d="M52 289Q59 331 106 386T222 442Q257 442 286 424T329 379Q371 442 430 442Q467 442 494 420T522 361Q522 332 508 314T481 292T458 288Q439 288 427 299T415 328Q415 374 465 391Q454 404 425 404Q412 404 406 402Q368 386 350 336Q290 115 290 78Q290 50 306 38T341 26Q378 26 414 59T463 140Q466 150 469 151T485 153H489Q504 153 504 145Q504 144 502 134Q486 77 440 33T333 -11Q263 -11 227 52Q186 -10 133 -10H127Q78 -10 57 16T35 71Q35 103 54 123T99 143Q142 143 142 101Q142 81 130 66T107 46T94 41L91 40Q91 39 97 36T113 29T132 26Q168 26 194 71Q203 87 217 139T245 247T261 313Q266 340 266 352Q266 380 251 392T217 404Q177 404 142 372T93 290Q91 281 88 280T72 278H58Q52 284 52 289Z" style="stroke-width:3;"></path></g></g><rect width="1004.7" height="60" x="120" y="220"></rect></g><g data-mml-node="mo" transform="translate(1841.7,0) translate(0 -0.5)"><path data-c="29" d="M35 1138Q35 1150 51 1150H56H69Q113 1113 153 1069T243 944T330 771T391 541T416 250T391 -40T330 -270T243 -443T152 -568T69 -649H56Q43 -649 39 -647T35 -637Q65 -607 110 -548Q283 -316 316 56Q324 133 324 251Q324 368 316 445Q278 877 48 1123Q36 1137 35 1138Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mi" transform="translate(2471.7,876.6) scale(0.707)"><path data-c="1D447" d="M40 437Q21 437 21 445Q21 450 37 501T71 602L88 651Q93 669 101 677H569H659Q691 677 697 676T704 667Q704 661 687 553T668 444Q668 437 649 437Q640 437 637 437T631 442L629 445Q629 451 635 490T641 551Q641 586 628 604T573 629Q568 630 515 631Q469 631 457 630T439 622Q438 621 368 343T298 60Q298 48 386 46Q418 46 427 45T436 36Q436 31 433 22Q429 4 424 1L422 0Q419 0 415 0Q410 0 363 1T228 2Q99 2 64 0H49Q43 6 43 9T45 27Q49 40 55 46H83H94Q174 46 189 55Q190 56 191 56Q196 59 201 76T241 233Q258 301 269 344Q339 619 339 625Q339 630 310 630H279Q212 630 191 624Q146 614 121 583T67 467Q60 445 57 441T43 437H40Z" style="stroke-width:3;"></path></g></g><g data-mml-node="mi" transform="translate(3019.5,0)"><path data-c="1D462" d="M21 287Q21 295 30 318T55 370T99 420T158 442Q204 442 227 417T250 358Q250 340 216 246T182 105Q182 62 196 45T238 27T291 44T328 78L339 95Q341 99 377 247Q407 367 413 387T427 416Q444 431 463 431Q480 431 488 421T496 402L420 84Q419 79 419 68Q419 43 426 35T447 26Q469 29 482 57T512 145Q514 153 532 153Q551 153 551 144Q550 139 549 130T540 98T523 55T498 17T462 -8Q454 -10 438 -10Q372 -10 347 46Q345 45 336 36T318 21T296 6T267 -6T233 -11Q189 -11 155 7Q103 38 103 113Q103 170 138 262T173 379Q173 380 173 381Q173 390 173 393T169 400T158 404H154Q131 404 112 385T82 344T65 302T57 280Q55 278 41 278H27Q21 284 21 287Z" style="stroke-width:3;"></path></g></g></g>',1),j=[D],L=t("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[t("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[t("msup",null,[t("mrow",{"data-mjx-texclass":"INNER"},[t("mo",{"data-mjx-texclass":"OPEN"},"("),t("mfrac",null,[t("mrow",null,[t("mi",null,"∂"),t("mi",null,"f")]),t("mrow",null,[t("mi",null,"∂"),t("mi",null,"x")])]),t("mo",{"data-mjx-texclass":"CLOSE"},")")]),t("mi",null,"T")]),t("mi",null,"u")])],-1),C=a('<p><strong>Backends &amp; AD Packages</strong></p><table><thead><tr><th style="text-align:left;">Supported Backends</th><th style="text-align:left;">Packages Needed</th></tr></thead><tbody><tr><td style="text-align:left;"><code>AutoZygote</code></td><td style="text-align:left;"><code>Zygote.jl</code></td></tr></tbody></table><div class="warning custom-block"><p class="custom-block-title">Warning</p><p>Gradient wrt <code>u</code> in the reverse pass is always dropped.</p></div><p><strong>Arguments</strong></p><ul><li><p><code>f</code>: The function to compute the jacobian of.</p></li><li><p><code>backend</code>: The backend to use for computing the VJP.</p></li><li><p><code>x</code>: The input to the function.</p></li><li><p><code>u</code>: An object of the same structure as <code>f(x)</code>.</p></li></ul><p><strong>Returns</strong></p><ul><li><code>v</code>: The Vector Jacobian Product.</li></ul><p><a href="https://github.com/LuxDL/Lux.jl/blob/baa2092615116194ed3be513dc2f6b8126877de0/src/helpers/autodiff.jl#L1" target="_blank" rel="noreferrer">source</a></p>',8),P=a('<br><h2 id="Batched-AD" tabindex="-1">Batched AD <a class="header-anchor" href="#Batched-AD" aria-label="Permalink to &quot;Batched AD {#Batched-AD}&quot;">​</a></h2><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="Lux.batched_jacobian" href="#Lux.batched_jacobian">#</a> <b><u>Lux.batched_jacobian</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">batched_jacobian</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(f, backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractADType</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Computes the Jacobian of a function <code>f</code> with respect to a batch of inputs <code>x</code>. This expects the following properties for <code>y = f(x)</code>:</p><ol><li><p><code>ndims(y) ≥ 2</code></p></li><li><p><code>size(y, ndims(y)) == size(x, ndims(x))</code></p></li></ol><p><strong>Backends &amp; AD Packages</strong></p><table><thead><tr><th style="text-align:left;">Supported Backends</th><th style="text-align:left;">Packages Needed</th></tr></thead><tbody><tr><td style="text-align:left;"><code>AutoForwardDiff</code></td><td style="text-align:left;"><code>ForwardDiff.jl</code></td></tr><tr><td style="text-align:left;"><code>AutoZygote</code></td><td style="text-align:left;"><code>Zygote.jl</code></td></tr></tbody></table><p><strong>Arguments</strong></p><ul><li><p><code>f</code>: The function to compute the jacobian of.</p></li><li><p><code>backend</code>: The backend to use for computing the jacobian.</p></li><li><p><code>x</code>: The input to the function. Must have <code>ndims(x) ≥ 2</code>.</p></li></ul><p><strong>Returns</strong></p><ul><li><code>J</code>: The Jacobian of <code>f</code> with respect to <code>x</code>. This will be a 3D Array. If the dimensions of <code>x</code> are <code>(N₁, N₂, ..., Nₙ, B)</code> and of <code>y</code> are <code>(M₁, M₂, ..., Mₘ, B)</code>, then <code>J</code> will be a <code>((M₁ × M₂ × ... × Mₘ), (N₁ × N₂ × ... × Nₙ), B)</code> Array.</li></ul><div class="danger custom-block"><p class="custom-block-title">Danger</p><p><code>f(x)</code> must not be inter-mixing the batch dimensions, else the result will be incorrect. For example, if <code>f</code> contains operations like batch normalization, then the result will be incorrect.</p></div><p><a href="https://github.com/LuxDL/Lux.jl/blob/baa2092615116194ed3be513dc2f6b8126877de0/src/helpers/autodiff.jl#L81-L114" target="_blank" rel="noreferrer">source</a></p></div><br><h2 id="Nested-2nd-Order-AD" tabindex="-1">Nested 2nd Order AD <a class="header-anchor" href="#Nested-2nd-Order-AD" aria-label="Permalink to &quot;Nested 2nd Order AD {#Nested-2nd-Order-AD}&quot;">​</a></h2><p>Consult the <a href="/v0.5.52/manual/nested_autodiff#nested_autodiff">manual page on Nested AD</a> for information on nested automatic differentiation.</p><hr class="footnotes-sep"><section class="footnotes"><ol class="footnotes-list"><li id="fn1" class="footnote-item"><p>Note that <code>ChainRules.jl</code> is not really an AD package, but we have first-class support for packages that use <code>rrules</code>. <a href="#fnref1" class="footnote-backref">↩︎</a></p></li><li id="fn2" class="footnote-item"><p>This feature is supported downstream, but we don&#39;t extensively test it to ensure that it works with Lux. <a href="#fnref2" class="footnote-backref">↩︎</a> <a href="#fnref2:1" class="footnote-backref">↩︎</a> <a href="#fnref2:2" class="footnote-backref">↩︎</a> <a href="#fnref2:3" class="footnote-backref">↩︎</a> <a href="#fnref2:4" class="footnote-backref">↩︎</a> <a href="#fnref2:5" class="footnote-backref">↩︎</a> <a href="#fnref2:6" class="footnote-backref">↩︎</a></p></li></ol></section>',8);function E(I,V,S,M,H,N){return s(),o("div",null,[r,t("div",d,[n,e(" "),c,e(" — "),h,e(". "),p,t("p",null,[e("Compute the Jacobian-Vector Product "),t("mjx-container",f,[(s(),o("svg",T,Q)),g]),e(". This is a wrapper around AD backends but allows us to compute gradients of jacobian-vector products efficiently using mixed-mode AD.")]),m]),b,t("div",x,[_,e(" "),k,e(" — "),y,e(". "),w,t("p",null,[e("Compute the Vector-Jacobian Product "),t("mjx-container",v,[(s(),o("svg",A,j)),L]),e(". This is a wrapper around AD backends but allows us to compute gradients of vector-jacobian products efficiently using mixed-mode AD.")]),C]),P])}const B=l(i,[["render",E]]);export{J as __pageData,B as default};
