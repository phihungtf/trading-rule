---
# try also 'default' to start simple
theme: apple-basic
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
layout: intro-image
image: /background.jpg
# some information about your slides, markdown enabled
title: Trading Rule Identification by CNN | Group 9
info: |
  ## Trading Rule Identification by CNN
  Group 9

  Learn more at [Google Colab](https://colab.research.google.com/drive/1vZx9P75a1Gmj0B90-0o28vHid54T9Mtx?usp=sharing)
# apply any unocss classes to the current slide
class: text-center
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# https://sli.dev/guide/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/guide/syntax#mdc-syntax
mdc: true
---

<div class="flex items-center justify-center space-x-5 mb-10 text-black">
  <div>
    <img src="/hcmus.png" alt="HCMUS" class="h-20">
  </div>
  <div class="flex flex-col space-y-1 items-start">
    <div class="text-xs">Vietnam National University Ho Chi Minh City</div>
    <div class="text-xs">University of Science</div>
    <div class="text-xs">Faculty of Information Technology</div>
    <div class="text-xs">Module: Advanced Topics in Software Development Technology</div>
  </div>
</div>

<h1 class="text-black">Trading Rule Identification</h1>

<h1 class="text-black">by CNN</h1>

<div class="flex justify-between mx-10 text-black text-shadow-md">
  <div class="flex flex-col items-start">
    <div>Author: Group 9</div>
    <div>20120454 - Lê Công Đắt</div>
    <div>20120489 - Võ Phi Hùng</div>
    <div>20120558 - Lưu Ngọc Quang</div>
    <div>20120582 - Trần Hữu Thành</div>
  </div>
  <div class="flex flex-col items-start">
    <div>Instructor:</div>
    <div>M.S. Trần Văn Quý</div>
    <div>M.S. Trần Duy Quang</div>
    <div>M.S. Đỗ Nguyên Kha</div>
  </div>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none text-black">
    <carbon:edit />
  </button>
  <a href="https://colab.research.google.com/drive/1vZx9P75a1Gmj0B90-0o28vHid54T9Mtx?usp=sharing" target="_blank" alt="Google Colab" title="Open in Google Colab"
    class="text-xl slidev-icon-btn opacity-50 !border-none text-black">
    <img src="/colab.png" width="30px"/>
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---

<style>
.toc {
  text-align: left;
  margin: 2em;
}

.toc-item {
  margin-bottom: 0.5em;
  font-size: 1.2em;
  color: #3498db;
  cursor: pointer;
  transition: all 0.3s ease;
}

.toc-item:hover {
  color: #1abc9c;
}

.toc-subitem {
  margin-left: 1em;
  font-size: 1.1em;
  color: #3498db;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-bottom: 0.5em;
}

.toc-subitem:hover {
  color: #27ae60;
}
</style>

# Table of Contents

<div class="toc">
  <div class="toc-item" @click="$slidev.nav.go(1)">1. Introduction</div>
  <div class="toc-item" @click="$slidev.nav.go(2)">2. Trading signals with technical indicators</div>
  <div class="toc-item" @click="$slidev.nav.go(3)">3. Data handling</div>
  <div class="toc-item" @click="$slidev.nav.go(4)">4. Benchmarking alternative models</div>
  <div class="toc-subitem" @click="$slidev.nav.go(5)">4.1. Benchmark 1 – simple trading rule</div>
  <div class="toc-subitem" @click="$slidev.nav.go(6)">4.2. Benchmark 2 – simple classification network</div>
  <div class="toc-item" @click="$slidev.nav.go(7)">5. Constructing a convolutional neural network</div>
  <div class="toc-item" @click="$slidev.nav.go(8)">6. Summary</div>
</div>

---
layout: intro-image-right
image: "/introduction.jpg"
---

<!-- Phần Introduction -->

# Introduction

---
src: ./pages/intro.md
---

---
layout: intro-image-right
image: "/signal.jpg"
---

<!-- Phần Trading signals with technical indicators -->

# Trading signals with technical indicators

---
src: ./pages/signal_mov1.md
---

---
src: ./pages/signal_mov2.md
---

---
src: ./pages/signal_rule.md
---

---
layout: intro-image-right
image: "/data.png"
---

<!-- Phần Data handling -->

# Data handling

---
src: ./pages/data_handling.md
---

---
layout: intro-image-right
image: "/ML.jpg"
---

<!-- Phần Benchmarking alternative models -->

# Benchmarking alternative models

---
src: ./pages/benchmarking.md
---

---
layout: quote
---

<!-- Phần Constructing a convolutional neural network -->
  <div class="flex justify-between items-center">
    <h1>Constructing a convolutional neural network</h1>
    <img src="/Ai-Neural-Network.gif" width="500px"></img>
  </div>

---
src: ./pages/cnn.md
---

---
layout: intro-image-right
image: "/summary.jpg"
---

<!-- Phần Summary -->

# Summary

---
src: ./pages/summary.md
---
