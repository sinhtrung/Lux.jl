import{_ as p,c as a,o as i,a2 as n,j as s,a as e}from"./chunks/framework.v7Shlql0.js";const O=JSON.parse('{"title":"Training a Neural ODE to Model Gravitational Waveforms","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/advanced/1_GravitationalWaveForm.md","filePath":"tutorials/advanced/1_GravitationalWaveForm.md","lastUpdated":null}'),t={name:"tutorials/advanced/1_GravitationalWaveForm.md"},l={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},h={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.339ex"},xmlns:"http://www.w3.org/2000/svg",width:"10.819ex",height:"1.658ex",role:"img",focusable:"false",viewBox:"0 -583 4782.1 733","aria-hidden":"true"},k={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},r={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.339ex"},xmlns:"http://www.w3.org/2000/svg",width:"2.008ex",height:"1.339ex",role:"img",focusable:"false",viewBox:"0 -442 887.6 592","aria-hidden":"true"},E={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},d={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.339ex"},xmlns:"http://www.w3.org/2000/svg",width:"2.008ex",height:"1.339ex",role:"img",focusable:"false",viewBox:"0 -442 887.6 592","aria-hidden":"true"},g={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},o={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"24.527ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 10840.9 1000","aria-hidden":"true"},C={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},Q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.117ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3587.6 1000","aria-hidden":"true"},f={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},y={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.049ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3557.6 1000","aria-hidden":"true"},v={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},c={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.138ex",height:"1.439ex",role:"img",focusable:"false",viewBox:"0 -442 503 636","aria-hidden":"true"},u={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},I={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"0"},xmlns:"http://www.w3.org/2000/svg",width:"2.378ex",height:"1.545ex",role:"img",focusable:"false",viewBox:"0 -683 1051 683","aria-hidden":"true"},m={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},F={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.054ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 466 453","aria-hidden":"true"},T={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},q={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.117ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3587.6 1000","aria-hidden":"true"},D={class:"MathJax",jax:"SVG",display:"true",style:{direction:"ltr",display:"block","text-align":"center",margin:"1em 0",position:"relative"}},b={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.566ex"},xmlns:"http://www.w3.org/2000/svg",width:"8.049ex",height:"2.262ex",role:"img",focusable:"false",viewBox:"0 -750 3557.6 1000","aria-hidden":"true"},V={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},B={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.439ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.138ex",height:"1.439ex",role:"img",focusable:"false",viewBox:"0 -442 503 636","aria-hidden":"true"},z={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},R={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"0"},xmlns:"http://www.w3.org/2000/svg",width:"2.378ex",height:"1.545ex",role:"img",focusable:"false",viewBox:"0 -683 1051 683","aria-hidden":"true"},U={class:"MathJax",jax:"SVG",style:{direction:"ltr",position:"relative"}},P={style:{overflow:"visible","min-height":"1px","min-width":"1px","vertical-align":"-0.025ex"},xmlns:"http://www.w3.org/2000/svg",width:"1.054ex",height:"1.025ex",role:"img",focusable:"false",viewBox:"0 -442 466 453","aria-hidden":"true"};function K(x,A,S,w,Z,X){return i(),a("div",null,[A[41]||(A[41]=n("",8)),s("p",null,[A[6]||(A[6]=e("We need a very crude 2-body path. Assume the 1-body motion is a newtonian 2-body position vector ")),s("mjx-container",l,[(i(),a("svg",h,A[0]||(A[0]=[n("",1)]))),A[1]||(A[1]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"r"),s("mo",null,"="),s("msub",null,[s("mi",null,"r"),s("mn",null,"1")]),s("mo",null,"−"),s("msub",null,[s("mi",null,"r"),s("mn",null,"2")])])],-1))]),A[7]||(A[7]=e(" and use Newtonian formulas to get ")),s("mjx-container",k,[(i(),a("svg",r,A[2]||(A[2]=[n("",1)]))),A[3]||(A[3]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("msub",null,[s("mi",null,"r"),s("mn",null,"1")])])],-1))]),A[8]||(A[8]=e(", ")),s("mjx-container",E,[(i(),a("svg",d,A[4]||(A[4]=[n("",1)]))),A[5]||(A[5]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("msub",null,[s("mi",null,"r"),s("mn",null,"2")])])],-1))]),A[9]||(A[9]=e(" (e.g. Theoretical Mechanics of Particles and Continua 4.3)"))]),A[42]||(A[42]=n("",2)),s("p",null,[A[12]||(A[12]=e("Next we define a function to perform the change of variables: ")),s("mjx-container",g,[(i(),a("svg",o,A[10]||(A[10]=[n("",1)]))),A[11]||(A[11]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mo",{stretchy:"false"},"("),s("mi",null,"χ"),s("mo",{stretchy:"false"},"("),s("mi",null,"t"),s("mo",{stretchy:"false"},")"),s("mo",null,","),s("mi",null,"ϕ"),s("mo",{stretchy:"false"},"("),s("mi",null,"t"),s("mo",{stretchy:"false"},")"),s("mo",{stretchy:"false"},")"),s("mo",{stretchy:"false"},"↦"),s("mo",{stretchy:"false"},"("),s("mi",null,"x"),s("mo",{stretchy:"false"},"("),s("mi",null,"t"),s("mo",{stretchy:"false"},")"),s("mo",null,","),s("mi",null,"y"),s("mo",{stretchy:"false"},"("),s("mi",null,"t"),s("mo",{stretchy:"false"},")"),s("mo",{stretchy:"false"},")")])],-1))])]),A[43]||(A[43]=n("",13)),s("mjx-container",C,[(i(),a("svg",Q,A[13]||(A[13]=[n("",1)]))),A[14]||(A[14]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("mi",null,"u"),s("mo",{stretchy:"false"},"["),s("mn",null,"1"),s("mo",{stretchy:"false"},"]"),s("mo",null,"="),s("mi",null,"χ")])],-1))]),s("mjx-container",f,[(i(),a("svg",y,A[15]||(A[15]=[n("",1)]))),A[16]||(A[16]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("mi",null,"u"),s("mo",{stretchy:"false"},"["),s("mn",null,"2"),s("mo",{stretchy:"false"},"]"),s("mo",null,"="),s("mi",null,"ϕ")])],-1))]),s("p",null,[A[23]||(A[23]=e("where, ")),s("mjx-container",v,[(i(),a("svg",c,A[17]||(A[17]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D45D",d:"M23 287Q24 290 25 295T30 317T40 348T55 381T75 411T101 433T134 442Q209 442 230 378L240 387Q302 442 358 442Q423 442 460 395T497 281Q497 173 421 82T249 -10Q227 -10 210 -4Q199 1 187 11T168 28L161 36Q160 35 139 -51T118 -138Q118 -144 126 -145T163 -148H188Q194 -155 194 -157T191 -175Q188 -187 185 -190T172 -194Q170 -194 161 -194T127 -193T65 -192Q-5 -192 -24 -194H-32Q-39 -187 -39 -183Q-37 -156 -26 -148H-6Q28 -147 33 -136Q36 -130 94 103T155 350Q156 355 156 364Q156 405 131 405Q109 405 94 377T71 316T59 280Q57 278 43 278H29Q23 284 23 287ZM178 102Q200 26 252 26Q282 26 310 49T356 107Q374 141 392 215T411 325V331Q411 405 350 405Q339 405 328 402T306 393T286 380T269 365T254 350T243 336T235 326L232 322Q232 321 229 308T218 264T204 212Q178 106 178 102Z",style:{"stroke-width":"3"}})])])],-1)]))),A[18]||(A[18]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"p")])],-1))]),A[24]||(A[24]=e(", ")),s("mjx-container",u,[(i(),a("svg",I,A[19]||(A[19]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D440",d:"M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z",style:{"stroke-width":"3"}})])])],-1)]))),A[20]||(A[20]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"M")])],-1))]),A[25]||(A[25]=e(", and ")),s("mjx-container",m,[(i(),a("svg",F,A[21]||(A[21]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D452",d:"M39 168Q39 225 58 272T107 350T174 402T244 433T307 442H310Q355 442 388 420T421 355Q421 265 310 237Q261 224 176 223Q139 223 138 221Q138 219 132 186T125 128Q125 81 146 54T209 26T302 45T394 111Q403 121 406 121Q410 121 419 112T429 98T420 82T390 55T344 24T281 -1T205 -11Q126 -11 83 42T39 168ZM373 353Q367 405 305 405Q272 405 244 391T199 357T170 316T154 280T149 261Q149 260 169 260Q282 260 327 284T373 353Z",style:{"stroke-width":"3"}})])])],-1)]))),A[22]||(A[22]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"e")])],-1))]),A[26]||(A[26]=e(" are constants"))]),A[44]||(A[44]=n("",14)),s("mjx-container",T,[(i(),a("svg",q,A[27]||(A[27]=[n("",1)]))),A[28]||(A[28]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("mi",null,"u"),s("mo",{stretchy:"false"},"["),s("mn",null,"1"),s("mo",{stretchy:"false"},"]"),s("mo",null,"="),s("mi",null,"χ")])],-1))]),s("mjx-container",D,[(i(),a("svg",b,A[29]||(A[29]=[n("",1)]))),A[30]||(A[30]=s("mjx-assistive-mml",{unselectable:"on",display:"block",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",overflow:"hidden",width:"100%"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},[s("mi",null,"u"),s("mo",{stretchy:"false"},"["),s("mn",null,"2"),s("mo",{stretchy:"false"},"]"),s("mo",null,"="),s("mi",null,"ϕ")])],-1))]),s("p",null,[A[37]||(A[37]=e("where, ")),s("mjx-container",V,[(i(),a("svg",B,A[31]||(A[31]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D45D",d:"M23 287Q24 290 25 295T30 317T40 348T55 381T75 411T101 433T134 442Q209 442 230 378L240 387Q302 442 358 442Q423 442 460 395T497 281Q497 173 421 82T249 -10Q227 -10 210 -4Q199 1 187 11T168 28L161 36Q160 35 139 -51T118 -138Q118 -144 126 -145T163 -148H188Q194 -155 194 -157T191 -175Q188 -187 185 -190T172 -194Q170 -194 161 -194T127 -193T65 -192Q-5 -192 -24 -194H-32Q-39 -187 -39 -183Q-37 -156 -26 -148H-6Q28 -147 33 -136Q36 -130 94 103T155 350Q156 355 156 364Q156 405 131 405Q109 405 94 377T71 316T59 280Q57 278 43 278H29Q23 284 23 287ZM178 102Q200 26 252 26Q282 26 310 49T356 107Q374 141 392 215T411 325V331Q411 405 350 405Q339 405 328 402T306 393T286 380T269 365T254 350T243 336T235 326L232 322Q232 321 229 308T218 264T204 212Q178 106 178 102Z",style:{"stroke-width":"3"}})])])],-1)]))),A[32]||(A[32]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"p")])],-1))]),A[38]||(A[38]=e(", ")),s("mjx-container",z,[(i(),a("svg",R,A[33]||(A[33]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D440",d:"M289 629Q289 635 232 637Q208 637 201 638T194 648Q194 649 196 659Q197 662 198 666T199 671T201 676T203 679T207 681T212 683T220 683T232 684Q238 684 262 684T307 683Q386 683 398 683T414 678Q415 674 451 396L487 117L510 154Q534 190 574 254T662 394Q837 673 839 675Q840 676 842 678T846 681L852 683H948Q965 683 988 683T1017 684Q1051 684 1051 673Q1051 668 1048 656T1045 643Q1041 637 1008 637Q968 636 957 634T939 623Q936 618 867 340T797 59Q797 55 798 54T805 50T822 48T855 46H886Q892 37 892 35Q892 19 885 5Q880 0 869 0Q864 0 828 1T736 2Q675 2 644 2T609 1Q592 1 592 11Q592 13 594 25Q598 41 602 43T625 46Q652 46 685 49Q699 52 704 61Q706 65 742 207T813 490T848 631L654 322Q458 10 453 5Q451 4 449 3Q444 0 433 0Q418 0 415 7Q413 11 374 317L335 624L267 354Q200 88 200 79Q206 46 272 46H282Q288 41 289 37T286 19Q282 3 278 1Q274 0 267 0Q265 0 255 0T221 1T157 2Q127 2 95 1T58 0Q43 0 39 2T35 11Q35 13 38 25T43 40Q45 46 65 46Q135 46 154 86Q158 92 223 354T289 629Z",style:{"stroke-width":"3"}})])])],-1)]))),A[34]||(A[34]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"M")])],-1))]),A[39]||(A[39]=e(", and ")),s("mjx-container",U,[(i(),a("svg",P,A[35]||(A[35]=[s("g",{stroke:"currentColor",fill:"currentColor","stroke-width":"0",transform:"scale(1,-1)"},[s("g",{"data-mml-node":"math"},[s("g",{"data-mml-node":"mi"},[s("path",{"data-c":"1D452",d:"M39 168Q39 225 58 272T107 350T174 402T244 433T307 442H310Q355 442 388 420T421 355Q421 265 310 237Q261 224 176 223Q139 223 138 221Q138 219 132 186T125 128Q125 81 146 54T209 26T302 45T394 111Q403 121 406 121Q410 121 419 112T429 98T420 82T390 55T344 24T281 -1T205 -11Q126 -11 83 42T39 168ZM373 353Q367 405 305 405Q272 405 244 391T199 357T170 316T154 280T149 261Q149 260 169 260Q282 260 327 284T373 353Z",style:{"stroke-width":"3"}})])])],-1)]))),A[36]||(A[36]=s("mjx-assistive-mml",{unselectable:"on",display:"inline",style:{top:"0px",left:"0px",clip:"rect(1px, 1px, 1px, 1px)","-webkit-touch-callout":"none","-webkit-user-select":"none","-khtml-user-select":"none","-moz-user-select":"none","-ms-user-select":"none","user-select":"none",position:"absolute",padding:"1px 0px 0px 0px",border:"0px",display:"block",width:"auto",overflow:"hidden"}},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("mi",null,"e")])],-1))]),A[40]||(A[40]=e(" are constants"))]),A[45]||(A[45]=n("",31))])}const M=p(t,[["render",K]]);export{O as __pageData,M as default};
