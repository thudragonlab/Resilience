import {
  Option
} from '@/templates/chart'

import {
  maxFontSize,
  middleFontSize
} from '@/styles/font.scss'
import {
  interval
} from 'd3'

const deepColor = '#AF86D8'
const softColor = '#7f5ac3'
const dashboardAreaColor = '#AF86D8'
const dashboardLineColor = '#7f5ac3'

export class HeatmapOption extends Option {
  constructor(title) {
    super()
    this.title = title
  }

  // getLegend() {
  //   return {
  //     top: '6%',
  //   }
  // }

  // getToolTip() {
  //   return {
  //     trigger: 'axis',
  //     axisPointer: {
  //       type: 'line'
  //     }
  //   }
  // }
  getXAxis() {
    return {
      type: 'category',
      splitArea: {
        show: true
      },
      axisLabel: {
        //x轴文字的配置
        show: true,
        interval: 0,//使x轴文字显示全
       }

      
    }
  }

  getYAxis() {
    return {
      type: 'category',
      splitArea: {
        show: true
      }
    }
  }

  getVisualMap() {
    return {
      // min: 0,
      // max: 10,
      calculable: true,
      max: 50,
      min: 0,
      right: 0,
      top:'middle',
      inRange: {
        color: ['#58b8cc', '#efd99c',  '#BD4E46'],
      },
      // bottom: '15%'
    }
  }

  getGrid(){
    return  {
      width: '80%',
      // left: 150
      right:100
    }
}
  
  getSeries() {
    return [{
      name: 'Punch Card',
      type: 'heatmap',
      label: {
        show: true,
        color:'#000'
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }]
  }

}