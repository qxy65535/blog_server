webpackJsonp([2],{FeBl:function(t,e){var n=t.exports={version:"2.5.3"};"number"==typeof __e&&(__e=n)},dvQz:function(t,e,n){"use strict";(function(t){var a=n("mvHQ"),o=n.n(a),i=n("mtWM"),r=n.n(i);e.a={name:"HandwtReco",data:function(){return{trBk:!1,popupColorShow:!1,popupColorType:"1",popupColorPosition:{x:"100",y:"100"},currentTool:"iconfont icon-maobi",currentToolNum:"1",tools:[{dataTool:"1",toolClass:"iconfont icon-maobi"}],selectedToolIndex:"0",lineWidth:"30",startx:999,starty:999,endx:0,endy:0,result:"",probability:""}},mounted:function(){var e=t(document).find("canvas"),n=e[0].getContext("2d");n.fillStyle="#fff",n.fillRect(0,0,e.width(),e.height())},methods:{painting:function(e){var n=this,a=t(e.target),o=a[0].getContext("2d"),i=(parseInt(this.getAttrValue(a,"width","px")),parseInt(this.getAttrValue(a,"height","px")),parseInt(a.position().left),parseInt(a.position().top),!1);t(document).on("mousedown","canvas",function(t){t.stopImmediatePropagation(),i=!0,o.beginPath(),o.lineCap="round",o.strokeStyle="#"+n.pageLineColor,o.lineWidth=n.lineWidth;var e=t.pageX-a.offset().left,r=t.pageY-a.offset().top;o.moveTo(e,r)}),t(document).on("mousemove","canvas",function(t){if(t.stopImmediatePropagation(),!i)return!1;var e=t.pageX-a.offset().left,r=t.pageY-a.offset().top;o.lineTo(e,r),o.stroke(),e<n.startx&&(n.startx=e),r<n.starty&&(n.starty=r),e>n.endx&&(n.endx=e),r>n.endy&&(n.endy=r)}),t(document).on("mouseup",function(t){t.stopImmediatePropagation(),i=!1,o.closePath()})},downloadImage:function(){var e=t(document).find("canvas"),n=(e[0].getContext("2d"),e[0].toDataURL("png").replace(this.fixType("png"),"image/octet-stream")),a=this;r()({url:"/recognition",method:"post",data:{imageData:n,x:Math.round(a.startx-30)>0?Math.round(a.startx-30):0,y:Math.round(a.starty-30)>0?Math.round(a.starty-30):0,endx:Math.round(a.endx+30)<300?Math.round(a.endx+30):300,endy:Math.round(a.endy+30)<300?Math.round(a.endy+30):300},contentType:"application/json"}).then(function(t){t.data.success?(a.result=t.data.message.predict,a.probability=o()(t.data.message.probability).replace(/,/g,"\n\t")):alert(t.data.message)}).catch(function(t){alert(t)})},fixType:function(t){return"image/"+(t=t.toLocaleLowerCase().replace(/jpg/i,"jpeg")).match(/png|jpeg|bmp|gif/)[0]},saveFile:function(t,e){var n=document.createElementNS("http://www.w3.org/1999/xhtml","a");n.href=t,n.download=e;var a=document.createEvent("MouseEvents");a.initMouseEvent("click",!0,!1,window,0,0,0,0,0,!1,!1,!1,!1,0,null),n.dispatchEvent(a)},clearScreen:function(){var e=t(document).find("canvas"),n=e[0].getContext("2d");n.fillStyle="#fff",n.fillRect(0,0,e.width(),e.height()),this.startx=999,this.starty=999,this.endx=0,this.endy=0,this.result="",this.probability=""},getAttrValue:function(t,e,n){return n=n||"",""+t.css(e).replace(n,"")}}}}).call(e,n("7t+N"))},jLiT:function(t,e,n){"use strict";Object.defineProperty(e,"__esModule",{value:!0});var a=n("dvQz"),o={render:function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{attrs:{id:"drawing-board-page"}},[n("header",[n("div",{staticClass:"current-tool"},[n("i",{class:t.currentTool}),t._v(" "),n("input",{directives:[{name:"model",rawName:"v-model",value:t.currentToolNum,expression:"currentToolNum"}],attrs:{type:"hidden",name:"currentTool"},domProps:{value:t.currentToolNum},on:{input:function(e){e.target.composing||(t.currentToolNum=e.target.value)}}})]),t._v(" "),t._m(0),t._v(" "),n("div",{staticClass:"downloadImage",on:{click:function(e){t.downloadImage()}}},[t._v("确认")]),t._v(" "),n("div",{staticClass:"clearScreen",on:{click:function(e){t.clearScreen()}}},[t._v("清屏")])]),t._v(" "),n("main",{staticClass:"clearfix"},[n("div",{staticStyle:{width:"300px",margin:"0 auto"}},[n("canvas",{attrs:{id:"drawing-board",width:"300px",height:"300px"},on:{mousedown:function(e){t.painting(e)}}})])]),t._v(" "),n("div",{staticClass:"clearfix",staticStyle:{width:"300px",margin:"30px auto"}},[n("pre",[t._v("结果："+t._s(t.result))]),t._v(" "),n("pre",[t._v("概率："+t._s(t.probability))])])])},staticRenderFns:[function(){var t=this.$createElement,e=this._self._c||t;return e("ul",{staticClass:"tool-info clearfix"},[e("li",{staticClass:"lineWidth"},[e("p",[this._v("手写数字识别")])])])}]};var i=function(t){n("nb8r")},r=n("VU/8")(a.a,o,!1,i,"data-v-3a46e50d",null);e.default=r.exports},mvHQ:function(t,e,n){t.exports={default:n("qkKv"),__esModule:!0}},nb8r:function(t,e){},qkKv:function(t,e,n){var a=n("FeBl"),o=a.JSON||(a.JSON={stringify:JSON.stringify});t.exports=function(t){return o.stringify.apply(o,arguments)}}});
//# sourceMappingURL=2.80b990da6f62067caf9f.js.map