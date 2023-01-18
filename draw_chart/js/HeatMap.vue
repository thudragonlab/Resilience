<template>
  <div class="test-container">
    <chart
      class="graph"
      :chartId="`line_122333`"
      height="60vh"
      :option="new HeatmapOption().getOption(['yAxis', 'xAxis', 'series', 'visualMap', 'grid'])"
      :sourceData="heatMapSourceDataAvg"
      :extraOption="{
        xAxis: {
          data: key_list,
          axisLabel: {
            //x轴文字的配置
            show: true,
            interval: 0, //使x轴文字显示全
          },
        },
        yAxis: {
          data: aspects,
        },
      }"
    ></chart>

    <!-- <chart
      chartId="one_month_line2"
      class="graph"
      height="80vh"
      :option="new LineTestOption('').getOption(['title', 'tooltip', 'xAxis', 'yAxis', 'legend', 'grid', 'color'])"
      :series="lineDataS"
    ></chart> -->
  </div>
</template>

<script>
// import testData from '@/../public/static/topo_graph_for_frontend/US_topo_graph.json'
// console.log(testData)
// import heatMapDataVar from '@/assets/temporaryData/var.json'
import heatMapDataAvg from '@/assets/temporaryData/med.json'
import lineData from '@/assets/temporaryData/2.json'
import { LineTestOption } from '@/templates/chart/lineTemplate'
import { HeatmapOption } from '@/templates/chart/heatmapTemplate'
import { ResilienceGraphOption } from '@/templates/chart/graphTemplate'

const DATA_PATH = '../../public/static/resilience/'
export default {
  name: 'Resilience',
  created() {},
  data() {
    return {
      GraphOption: ResilienceGraphOption,
      HeatmapOption,
      LineTestOption,
      // heatMapDataVar,
      heatMapDataAvg,
      lineData,
      // ,'NP','PK'
      key_list: [],
      heatMapSourceDataAvg: [],
      heatMapSourceDataVar: [],
      lineDataS: [],
      ccList: [
        'LV',
        'HU',
        'CL',
        'SK',
        'PT',
        'FI',
        'CO',
        'TR',
        'PH',
        'NG',
        'DK',
        'MX',
        'LU',
        'KH',
        'TW',
        'TH',
        'KR',
        'NO',
        'MY',
        'CN',
        'BG',
        'IR',
        'IE',
        'AR',
        'RO',
        'NZ',
        'CZ',
        'BD',
        'SE',
        'UA',
        'SG',
        'HK',
        'ES',
        'AT',
        'JP',
        'IN',
        'ID',
        'CH',
        'PL',
        'NL',
        'IT',
        'FR',
        'CA',
        'ZA',
        'AU',
        'GB',
        'DE',
        'US',
        'RU',
        'BR',
      ],
      aspects: [
        // 'domain-toposcope_hidden',
        // 'user-toposcope_hidden',
        // 'basic-toposcope_hidden',
        // 'domain-toposcope',
        // 'user-toposcope',
        // 'basic-toposcope',
        // 'domain-toposcope_hidden.200',
        // 'domain-toposcope_hidden.50',
        // 'domain-toposcope_hidden.45',
        // 'domain-toposcope_hidden.40',
        // 'domain-toposcope_hidden.35',
        // 'domain-toposcope_hidden.30',
        // 'domain-toposcope_hidden.25',
        // 'domain-toposcope_hidden.20',
        // 'domain-toposcope_hidden.15',
        // 'domain-toposcope_hidden.10',
        // 'domain-toposcope_hidden.5',
        'domain-toposcope_hidden',
        // 'user-toposcope_hidden.200',
        // 'user-toposcope_hidden.50',
        // 'user-toposcope_hidden.45',
        // 'user-toposcope_hidden.40',
        // 'user-toposcope_hidden.35',
        // 'user-toposcope_hidden.30',
        // 'user-toposcope_hidden.25',
        // 'user-toposcope_hidden.20',
        // 'user-toposcope_hidden.15',
        // 'user-toposcope_hidden.10',
        // 'user-toposcope_hidden.5',
        'user-toposcope_hidden',
        // 'basic-toposcope_hidden.200',
        // 'basic-toposcope_hidden.50',
        // 'basic-toposcope_hidden.45',
        // 'basic-toposcope_hidden.40',
        // 'basic-toposcope_hidden.35',
        // 'basic-toposcope_hidden.30',
        // 'basic-toposcope_hidden.25',
        // 'basic-toposcope_hidden.20',
        // 'basic-toposcope_hidden.15',
        // 'basic-toposcope_hidden.10',
        // 'basic-toposcope_hidden.5',
        'basic-toposcope_hidden',
        // 'domain-toposcope.200',
        // 'domain-toposcope.50',
        // 'domain-toposcope.45',
        // 'domain-toposcope.40',
        // 'domain-toposcope.35',
        // 'domain-toposcope.30',
        // 'domain-toposcope.25',
        // 'domain-toposcope.20',
        // 'domain-toposcope.15',
        // 'domain-toposcope.10',
        // 'domain-toposcope.5',
        'domain-toposcope',
        // 'user-toposcope.200',
        // 'user-toposcope.50',
        // 'user-toposcope.45',
        // 'user-toposcope.40',
        // 'user-toposcope.35',
        // 'user-toposcope.30',
        // 'user-toposcope.25',
        // 'user-toposcope.20',
        // 'user-toposcope.15',
        // 'user-toposcope.10',
        // 'user-toposcope.5',
        'user-toposcope',
        // 'basic-toposcope.200',
        // 'basic-toposcope.50',
        // 'basic-toposcope.45',
        // 'basic-toposcope.40',
        // 'basic-toposcope.35',
        // 'basic-toposcope.30',
        // 'basic-toposcope.25',
        // 'basic-toposcope.20',
        // 'basic-toposcope.15',
        // 'basic-toposcope.10',
        // 'basic-toposcope.5',
        'basic-toposcope',
        // 'domain-problink.200',
        // 'domain-problink.50',
        // 'domain-problink.45',
        // 'domain-problink.40',
        // 'domain-problink.35',
        // 'domain-problink.30',
        // 'domain-problink.25',
        // 'domain-problink.20',
        // 'domain-problink.15',
        // 'domain-problink.10',
        // 'domain-problink.5',
        'domain-problink',
        // 'user-problink.200',
        // 'user-problink.50',
        // 'user-problink.45',
        // 'user-problink.40',
        // 'user-problink.35',
        // 'user-problink.30',
        // 'user-problink.25',
        // 'user-problink.20',
        // 'user-problink.15',
        // 'user-problink.10',
        // 'user-problink.5',
        'user-problink',
        // 'basic-problink.200',
        // 'basic-problink.50',
        // 'basic-problink.45',
        // 'basic-problink.40',
        // 'basic-problink.35',
        // 'basic-problink.30',
        // 'basic-problink.25',
        // 'basic-problink.20',
        // 'basic-problink.15',
        // 'basic-problink.10',
        // 'basic-problink.5',
        'basic-problink',
        // 'domain-asRank.200',
        // 'domain-asRank.50',
        // 'domain-asRank.45',
        // 'domain-asRank.40',
        // 'domain-asRank.35',
        // 'domain-asRank.30',
        // 'domain-asRank.25',
        // 'domain-asRank.20',
        // 'domain-asRank.15',
        // 'domain-asRank.10',
        // 'domain-asRank.5',
        'domain-asRank',
        // 'user-asRank.200',
        // 'user-asRank.50',
        // 'user-asRank.45',
        // 'user-asRank.40',
        // 'user-asRank.35',
        // 'user-asRank.30',
        // 'user-asRank.25',
        // 'user-asRank.20',
        // 'user-asRank.15',
        // 'user-asRank.10',
        // 'user-asRank.5',
        'user-asRank',
        // 'basic-asRank.200',
        // 'basic-asRank.50',
        // 'basic-asRank.45',
        // 'basic-asRank.40',
        // 'basic-asRank.35',
        // 'basic-asRank.30',
        // 'basic-asRank.25',
        // 'basic-asRank.20',
        // 'basic-asRank.15',
        // 'basic-asRank.10',
        // 'basic-asRank.5',
        'basic-asRank',
      ].reverse(),
      useTone: true,
      searchInput: '',
      aspectsOfEchart: [
        {
          name: 'connectivity',
          data: [],
          showLineChart: false,
          echartLoading: false,
        },
        {
          name: 'user',
          data: [],
          showLineChart: false,
          echartLoading: false,
        },
        {
          name: 'domain',
          data: [],
          showLineChart: false,
          echartLoading: false,
        },
      ],
      selectedCC: 'AU',
      graphLink: [],
      graphSerise: [],
      graphDataCone: {},
      showTopo: true,
      maxCount: 10,
      AsAmount: 0,
      LinkAmount: 0,
      AsAmountOnGraph: 0,
      LinkAmountOnGraph: 0,
    }
  },
  props: {},
  mounted() {
    this.processHeatMapData(heatMapDataAvg, 'heatMapSourceDataAvg')
    // this.processHeatMapData(heatMapDataVar, 'heatMapSourceDataVar')
    this.processLine()
  },
  methods: {
    processLine() {
      this.lineDataS = lineData.map(item => {
        item.label = {
          formatter: param => {
            return param.seriesName
          },
        }

        // console.log(, item.name)
        const length = new Set(item.data.map(i => i.value[1])).size
        if (length === 1) {
          return
        }
        return item
      })
    },
    processHeatMapData(heatMapData, dataName) {
      const heatMapSourceData = []
      const value_list = Object.values(heatMapData).sort((a, b) => {
        // return eval(Object.values(a).join('+')) - eval(Object.values(b).join('+'))
        return a['basic-asRank'] - b['basic-asRank']
      })
      console.log(heatMapData)
      this.key_list = Object.keys(heatMapData).sort((a, b) => {
        // return eval(Object.values(heatMapData[a]).join('+')) - eval(Object.values(heatMapData[b]).join('+'))
        return heatMapData[a]['basic-asRank'] - heatMapData[b]['basic-asRank']
      })
      // this.key_list = this.ccList
      console.log(this.key_list)
      value_list.forEach((value, index) => {
        this.aspects.forEach((aspect, idx) => {
          if (this.ccList.includes(this.key_list[index])) {
            heatMapSourceData.push([index, idx, value[aspect]])
          }
          // if (index === 1) {
          //   console.log([index, idx, value[aspect]], value, aspect, this.key_list[index])
          // }
        })
        // Object.keys(value).forEach((k, idx) => {
        //   this.aspects[idx] = k
        //   heatMapSourceData.push([index, idx, value[k]])
        // })
        // heatMapSourceData.push([index, 0, value['0']])
        // heatMapSourceData.push([index, 1, value['1']])
        // heatMapSourceData.push([index, 2, value['2']])
        // heatMapSourceData.push([index, 3, value['3']])
        // heatMapSourceData.push([index, 4, value['4']])
        // heatMapSourceData.push([index, 5, value['5']])
        // heatMapSourceData.push([index, 6, value['6']])
        // heatMapSourceData.push([index, 7, value['7']])
        // heatMapSourceData.push([index, 8, value['8']])
        // heatMapSourceData.push([index, 9, value['9']])
        // heatMapSourceData.push([index, 10, value['10']])
        // heatMapSourceData.push([index, 11, value['11']])
        // heatMapSourceData.push([index, 12, value['12']])
      })
      this[dataName] = heatMapSourceData
    },
  },
}
</script>

<style lang="scss" scoped>
.title {
  font-size: var(--max-font-size);
  background-color: #fff;
  border-radius: 10px;
}

.test-container {
  padding: 22px;
  background-color: #f0f0f0;
  // height: 100%;
}

.cc_row {
  margin-left: 0 !important;
  margin-right: 0 !important;
  padding: 10px 0;
  background-color: #fff;
  border-radius: 10px;
  overflow: auto;
  // scrollbar-width: none;
  // -ms-overflow-style: none;
}

// .cc_row::-webkit-scrollbar {
//   display: none;
// }

// .cc-col {
//   display: flex;
//   // justify-content: flex-end;
//   // margin: 0 10px;
//   // padding: 0 10px;
// }

.cc {
  color: #fff;
  font-weight: bold;
  background-image: linear-gradient(to right, #7f5ac3, #af86d8);
  // width: 80%;
  width: 50px;
  padding: 0 15px;
  margin: 0 10px;
  height: 30px;
  line-height: 30px;
  border-radius: 5px;
  box-shadow: 5px 4px 13px -7px #000;
  cursor: pointer;
}

.cc:hover {
  opacity: 0.8;
}
.cc:focus,
.cc:active {
  box-shadow: -2px 4px 11px -7px #000 inset;
  // color:red
}
.line,
.graph {
  background-color: #fff;
  // border-radius: 10px;
}

.show-row {
  animation-name: fade-in;
  animation-duration: 0.5s;
}

.show-line-chart {
  animation-name: fade-in;
  animation-duration: 1s;
}

.switch-box {
  display: flex;
  justify-content: center;
  align-items: center;
}
.cone {
  text-align: left;
  z-index: 2;
  background-color: #fff;
  // position: absolute;
  // left: 20px;
  // top: 20px;
  animation-name: fade-in;
  animation-duration: 1s;
}
@keyframes fade-in {
  0% {
    opacity: 0;
    transform: translate(0, 10px);
  }

  100% {
    opacity: 1;
  }
}

.as-statistic {
  position: absolute;
  margin-top: 20px;
  z-index: 2;
  font-size: var(--middle-font-size);
  text-align: left;
  font-weight: bold;
  padding: 5px;
}

.grid-box {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-template-rows: repeat(3, 1fr);
  padding-top: 20px;
  background-color: #f0f0f0;
}

.topo-graph {
  grid-column-start: 1;
  grid-column-end: 5;
  grid-row-start: 1;
  grid-row-end: 4;
}
.clicked {
  color: #fff;
  background-image: linear-gradient(to right, #ee8b76, #e29a8b);
}
</style>
