<!DOCTYPE html>
<html>

<head>

<title>OpenGradient Intelligence Terminal</title>

<style>

body{
background:#05070d;
color:#00f2ff;
font-family:monospace;
margin:0;
}

.header{
text-align:center;
padding:40px;
font-size:36px;
}

.terminal{
max-width:900px;
margin:auto;
background:#0b0f1a;
padding:20px;
border-radius:10px;
}

.model{
border-bottom:1px solid #1a2333;
padding:18px;
}

.name{
font-size:20px;
color:white;
}

.description{
color:#9bb3c9;
margin-top:5px;
}

.likes{
color:#00f2ff;
font-size:13px;
}

button{
margin-top:10px;
background:#00f2ff;
border:none;
padding:6px 12px;
cursor:pointer;
border-radius:6px;
}

.code{
display:none;
background:#04060c;
padding:10px;
margin-top:10px;
white-space:pre;
border-radius:6px;
}

</style>

</head>

<body>

<div class="header">
OpenGradient Intelligence Terminal
</div>

<div class="terminal">

{% for m in models %}

<div class="model">

<div class="name">{{m.name}}</div>

<div class="description">{{m.description}}</div>

<div class="likes">❤ {{m.likes}} likes</div>

<button onclick="generate('{{m.name}}','code{{loop.index}}')">
Generate SDK Command
</button>

<div class="code" id="code{{loop.index}}"></div>

</div>

{% endfor %}

</div>

<script>

function generate(model,id){

let code=`pip install opengradient

import opengradient as og

llm = og.LLM(private_key="YOUR_PRIVATE_KEY")

response = llm.chat(
model="${model}",
messages=[{"role":"user","content":"Hello"}]
)

print(response)
`

let box=document.getElementById(id)

box.style.display="block"

box.innerText=code

}

</script>

</body>

</html>
