<template>
  <div
    :id="chartId"
    class="echart"
    :ref="chartId"
    :class="className"
    :style="{height, width}"
  />
</template>

<script>
import * as echarts from 'echarts'

/**
 * 基于Echart的组件
 * @author pzd
 */
let resizeTimer
window.addEventListener('resize', () => {
  if (resizeTimer) {
    clearTimeout(resizeTimer)
  }
  resizeTimer = setTimeout(() => {
    if(global.instanceList){
      global.instanceList.forEach(echartInstance => echartInstance.resize())
    }
  }, 500)
})
export default {
  name: 'Chart',
  props: {
    /**
     * Echart 模版，
     */
    option: {
      type: Object,
      default: () => {
        return {}
      },
    },
    /**
     * series中的data字段
     *
     * 该字段支持动态绑定
     *
     * 注意：仅限只有一个series元素时使用，比如Bar、Pie
     */
    sourceData: {
      type: Array,
      default: undefined,
    },
    /**
     * series字段
     *
     * 该字段支持动态绑定
     *
     */
    series: {
      type: Array,
      default: undefined,
    },
    /**
     * Option中其他需要动态变化的选项
     *
     * 该字段仅在有source或者series时刷新
     */
    extraOption: {
      type: Object,
      default: () => {
        return {}
      },
    },
    /**
     *  Echart对象额外的类名
     */
    extraSeriesOption: {
      type: Object,
      default: () => {
        return {}
      },
    },

    className: {
      type: String,
      default: 'chart',
    },
    /**
     *  Echart对象渲染DOM id
     *
     *  注意：此字段必须唯一
     */
    chartId: {
      type: String,
      default: 'main',
    },
    /**
     *  Echart对象渲染DOM宽度
     */
    width: {
      type: String,
      default: '100%',
    },
    /**
     *  Echart对象渲染DOM高度
     */
    height: {
      type: String,
      default: '100%',
    },
    /**
     * Echart绑定事件列表
     */
    EventList: {
      type: Object,
      default: () => {},
    },
    /**
     * Echart 刷新series需要执行的额外操作
     *
     * 通常和EventList配合使用
     */
    watchSeries: {
      type: Function,
      default: () => {},
    },
    replaceMerge:{
      type:Array,
      default:() => {
        return []
      }
    }
  },
  data() {
    return {
      chart: undefined,
      loading: false,
      scrollNowIndex: 0,
      scrollToIndex: 0,
    }
  },
  watch: {
    option() {},
    sourceData(newV) {
      if (!this.chart) {
        this.drawChart(Object.assign(this.option, this.extraOption))
      }
      const serise = [
        Object.assign(this.chart.getOption().series[0],{
          data: newV,
        })
        ,
      ]

      if (newV.length !== 0) {
        this.reDraw({
          series: serise,
          ...this.extraOption,
        })
        // 给了个[],但是this.series有值
        // } else if (newV.length === 0 && this.series && this.series.length !== 0) {
        //   return;
      } else {
        this.loading = true
      }
    },
    series: {
      handler(newV, oldV) {
        if (!this.chart) {
          this.chart = this.initChart()
        }
        if (this.watchSeries) {
          this.watchSeries(newV, oldV, this)
        }
        if (newV.length !== 0) {
          this.reDraw({
            ...this.extraOption,
            ...this.option,
            series: newV,
          })
          // } else if (newV.length === 0 && this.sourceData && this.sourceData.length !== 0) {
          //   return;
        } else {
          this.loading = true
        }
      },
    },
    extraOption: {
      handler(newV) {
        if (
          (this.series && this.series.length !== 0) ||
          (this.sourceData && this.sourceData.length !== 0)
        ) {
          this.reDraw(newV)
        } else if (
          // 如果数据从dataset中来
          newV.dataset &&
          newV.dataset.length
        ) {
          this.reDraw(newV)
        }
      },
    },
    loading: {
      handler(newV, oldV) {
        if (newV) {
          this.chart.showLoading({
            text: 'Empty Data',
            textColor: '#8a8e91',
            showSpinner: false,
            maskColor: 'rgba(255, 255, 255, 0.8)',
          })
        } else {
          this.chart.hideLoading()
        }
      },
    },
  },

  mounted() {
    this.$nextTick(() => {
      this.loading = true
      if (this.option.series && this.sourceData && this.sourceData.length !== 0) {
        this.option.series[0].data = this.sourceData
        this.loading = false
      }
      if (this.series && this.series.length !== 0) {
        this.option.series = this.series
        this.loading = false
      }

      if (
        this.extraOption &&
        this.extraOption.dataset &&
        this.extraOption.dataset !== 0
      ) {
        this.loading = false
      }

      this.drawChart(Object.assign(this.option, this.extraOption))
      this.AddChartEventListeners()
    })
  },
  methods: {
    AddChartEventListeners() {
      if (this.EventList) {
        const eventKeys = Object.keys(this.EventList)
        const that = this

        eventKeys.forEach(key => {
          if (this.chart) {
            this.chart.on(key, function (e) {
              that.EventList[key](e, that.chart, that)
            })
          }
        })
      }
    },
    initChart() {
      let chartDom = this.$refs[this.chartId]
      if (!global.instanceList) {
        global.instanceList = []
      }
      const echartInstance = echarts.init(chartDom)
      global.instanceList.push(echartInstance)
      return echartInstance
    },
    drawChart(option) {
      this.chart = this.initChart()
      this.chart.setOption(option)
    },
    setChartOption(option, notMerge = false) {
      this.chart.setOption(option, {notMerge, replaceMerge:this.replaceMerge})
      this.loading = false
    },
    registerMapByGeo(mapName, geoJson) {
      echarts.registerMap(mapName, geoJson)
    },
    getChartIntance() {
      return this.chart
    },
    reDraw(option) {
      if (option) {
        this.setChartOption(option)
      }
    },
  },
}
</script>
