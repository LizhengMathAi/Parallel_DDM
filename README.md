<h1><b>Introduction</b></h1>
Domain decomposition methods(DDM) are wildly used in computer vision, numerical analysis, and numerical partial differential equations.
<font color="red">
    Traditional DDM base on geometry. however, when dimension or the number of node in a large scale, this problem become extremely difficult. But in my method, I convert this problem into a numerical optimization task. Thus, I can optimize the mesh by using Gradient Descent!
</font><br>
<ol>
    <li>This is an original algorithm for optimizing the positions of nodes in Domain Decomposition Task.
    <li>This method can be used in situations of any dimension.
    <li>Using this method, you can define your custom boundary nodes(must be the vertices of any convex hull) and any number of random inner nodes.
     <li>The process of optimizing inner nodes base on numerical optimization methods, thus this method is efficient and parallel.
</ol>
<b>To learn more about this method and run it by yourself</b>, click here <i><a href="http://www.li-zheng.net:8000/algorithms/domain_decomposition_methods.html">Domain Decomposition Methods</a></i>

<h1><b>Requirements</b></h1>
numpy==1.19.2<br>
scipy==1.5.2

<h1><b>Demo</b></h1>
2D demo 
<img src="https://github.com/LizhengMathAi/Parallel_DDM/blob/main/src/origin2d.png" />
<img src="https://github.com/LizhengMathAi/Parallel_DDM/blob/main/src/target2d.png" />
2D demo 
<img src="https://github.com/LizhengMathAi/Parallel_DDM/blob/main/src/origin3d.png" />
<img src="https://github.com/LizhengMathAi/Parallel_DDM/blob/main/src/target3d.png" />

<h1><b>Relevant Knowledge</b></h1>
<ul>
    <li>Function approximation
    <li>Numerical differentiation
    <li>Gradient Descent
    <li>Delaunay triangulation
    <li>CSR matrix
    <li>Dijkstra's algorithm
    <li>QR decomposition
    <li>Projection
</ul>
