<template>
	<v-app>
		<div class="graphBoxes">
        <div class="threeDGraphBox">
          <div class="titleGraphBox">3D Force Graph</div>
          <div class="moveThreeDGraphBox">
            <div id="3d-graph"></div>
          </div>
        </div>
        <div class="splitLine">
        </div>
        <div class="twoDGraphBox">
          <!-- <div class="titleGraphBox">2D Graph</div> -->
          <div id="2d-graph"></div>
        </div>
      </div>
	</v-app>
</template>


<style scoped>

  .graphBoxes {
    display:flex;
  }
  .threeDGraphBox {
    flex:2;
    position: relative;
    width: 100vh;
    height: 100vh;
    overflow:hidden;
  }
  .moveThreeDGraphBox {
    position:relative;
    left: 0%;
  }
  .titleGraphBox {
    position:absolute;
    margin-top: 50px;
    margin-left: 50px;
    z-index: 999;
    font-size: 25px;
  }
  .twoDGraphBox {
    flex:1;
    background-color: white;
    width: 100vh;
    height: 100vh;
    z-index: 1;
  }

  .splitLine {
    background-color: black;
    width: 1px;
    height: 100vh;
  }

</style>


<script lang="ts">
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
} from 'chart.js'

import { Scatter } from 'vue-chartjs'
import * as chartConfig from './chartConfig.js'
import ForceGraph3D from "3d-force-graph";
import Plotly from 'plotly.js-dist-min'
// import ForceGraph from 'force-graph';
import SpriteText from "three-spritetext";
import jsonData2 from "../data/data_clear.json"; // 4K
// import jsonData from "../data/data_cluster_3950_all_1281.json";
// import jsonData from "../data/data_cluster_3750_all_128.json";
// import jsonData from "../data/data_cluster_3_9k_top1.json";
// import jsonData from "../data/data_cluster_3_5k_top_k_centr_2d.json";
// import jsonData from "../data/data_cluster_3_5k_top_k_centr.json";
// import jsonData from "../data/data_cluster_3k.json";
// import jsonData from "../data/data_cluster_2k_all.json";
// import jsonData from "../data/data_cluster_2k_top1.json";
// import jsonData from "../data/data_cluster_50_all_2d.json";
// import jsonData from "../data/data_cluster_50_all_312.json";
// import jsonData from "../data/data_cluster_3500_all_128.json";
// import jsonData from "../data/data_cluster_3000_all_128.json";
// import jsonData from "../data/data_cluster_2500_all_128.json";
import jsonData from "../data/data_cluster_2500_all_128_with_name1.json";
// import jsonData from "../data/data_cluster_2500_all_128_with_top_words.json";



// and big black nothingness began to spin
// a system of cells interlinked within cells interlinked within cells interlinked within one stem
// and dreadfully distinct against the dark, a tall white fountain played

// кроваво красное ничто вдруг быстро начало вертеться
// система полносвязных клеток внутри другой системы полносвязных клеток внутри другой системы полносвязных клеток внутри одного объединяющего стебля
// до жути ясно ввысь белы

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend)

export default {
  name: 'App',
  components: {
    Scatter
  },
  data() {
    return chartConfig
  },
  mounted() {
        // 3Д граф // смотреть тут https://github.com/vasturiano/3d-force-graph
        // const graph = ForceGraph3D({ controlType: "orbit" })(document.getElementById("3d-graph"));


        const dddgraph = ForceGraph3D({ controlType: "orbit" })(document.getElementById("3d-graph"));
        // const ddgraph = ForceGraph();
  
        // const data = {
        //   nodes: [
        //     { id: 'node1', x: 0, y: 0, z: 0, fixed: true  },  // Fixed position for node1
        //     { id: 'node2' },
        //     { id: 'node3' }
        //   ],
        //   links: [
        //     // { source: 'node1', target: 'node2' },
        //     // { source: 'node1', target: 'node3' },
        //     // { source: 'node2', target: 'node3' }
        //   ]
        // };


        var trace1 = {

          x: [1, 2, 3, 4, 5],

          y: [1, 6, 3, 6, 1],

          mode: 'markers',

          type: 'scatter',

          name: 'Team A',

          text: ['A-1', 'A-2', 'A-3', 'A-4', 'A-5'],

          marker: { size: 12 }

          };


          var trace2 = {

          x: [1.5, 2.5, 3.5, 4.5, 5.5],

          y: [4, 1, 7, 1, 4],

          mode: 'markers',

          type: 'scatter',

          name: 'Team B',

          text: ['B-a', 'B-b', 'B-c', 'B-d', 'B-e'],

          marker: { size: 12 }

          };


          var data = [ trace1, trace2 ];


        var layout = {
          scattermode: 'group',
          xaxis: {title: 'Country'},
          yaxis: {title: 'Medals'},
          width: 700,
          // plot_bgcolor:"black",
          height: 1100,
          scattergap: 0.7
        };


        Plotly.newPlot('2d-graph', data, layout);



        // ddgraph(document.getElementById("2d-graph"))
        //   // .graphData(jsonData2)
        //   .graphData(data)
        //   .width(510)
        //   // .height(750)
        //   .backgroundColor('#000000')
        //   .onNodeDragEnd(node => {
        //     node.fx = node.x;
        //     node.fy = node.y;
        //     node.fz = node.z;
        //   })


        // const linkForce = graph
        //   .d3Force('link')
        //   .distance(_ => settings.Distance);

        const highlightNodes = new Set();
        const highlightLinks = new Set();
        let hoverNode = null;

        dddgraph
          // .dagMode("td")
          .width(1050)
          // .height(100)
          .graphData(jsonData)
          .nodeLabel('id')
          .linkOpacity(1)
          .linkWidth(1)
				  .nodeColor("")
          .linkDirectionalParticles((link) => (highlightLinks.has(link) ? 4 : 0))
				  .linkDirectionalParticleWidth(16)
          .enableNodeDrag(false)
          .linkDirectionalArrowResolution(2)
          .nodeResolution(1)
          .nodeRelSize(16)
          .backgroundColor('#FFFFFF')
          // .onNodeDragEnd(node => {
          //   node.fx = node.x;
          //   node.fy = node.y;
          //   node.fz = node.z;
          // })
          .nodeLabel((node) => node.text)
          // .nodeThreeObject((node) => {
          //   const sprite = new SpriteText(node.text);
          //   sprite.material.depthWrite = true; // ma	ke sprite background transparent
          //   sprite.color = "White";
          //   sprite.textHeight = 4;
          //   return sprite;
          // })
          .onNodeClick((node) => {
            const distance = 250;
            const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
            dddgraph.cameraPosition({ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, node, 3000);
  				});

  }
}
</script>
