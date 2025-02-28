# MA-GTS
MA-GTS: A Multi-Agent Framework for Solving Complex Graph Problems in Real-World Applications
## Abstract
Graph-theoretic problems arise in real-world applications like logistics, communication networks, and traffic optimization. These problems are often complex, noisy, and irregular, posing challenges for traditional algorithms. Large language models (LLMs) offer potential solutions but face challenges, including limited accuracy and input length constraints. To address these challenges, we propose MA-GTS (Multi-Agent Graph Theory Solver), a multi-agent framework that decomposes these complex problems through agent collaboration. MA-GTS maps the implicitly expressed text-based graph data into clear, structured graph representations and dynamically selects the most suitable algorithm based on problem constraints and graph structure scale. This approach ensures that the solution process remains efficient and the resulting reasoning path is interpretable. We validate MA-GTS using the G-REAL dataset, a real-world-inspired graph theory dataset we created. Experimental results show that MA-GTS outperforms state-of-the-art approaches in terms of efficiency, accuracy, and scalability, with strong results across multiple benchmarks (G-REAL 94.2%, GraCoRe 96.9%, NLGraph 98.4%).
## Framework of MA-GTS

 <p align="center">
  <img width="65%" src='./PIC/framework.png' />
</p>

## Pipeline of MA-GTS

 <p align="center">
  <img width="65%" src='./PIC/pipeline.png' />
</p>