Компилировать надо файлы dissertation.tex (для диссертации), synopsis.tex (для автореферата), report.tex (для научного доклада).

Для библиографии используется BibLaTeX, а не BibTeX, поэтому может потребоваться соответствующая настройка TeX-редактора. Я пользовался TeXworks под Windows и добавил инструмент вёрстки "pdfLaTeX+MakeIndex+BibLaTeX", запускающий батник biblatexify.bat с параметрами $fullname и $basename.

Основной текст содержится в папке text, как это ни удивительно. Информация для титульных листов указывается в text/title_data.tex, сами титульные листы генерируются в formatted_text/dissertation_title.tex, formatted_text/synopsis_title.tex и formatted_text/report_title.tex. Для научного доклада некоторая информация вставлена руками в report_title.tex и не содержится в title_data.tex.

Введение содержится в файлах text/introduction.tex и text/synopsis_intro.tex, которые потом инпутятся в formatted_text/intro_formatted.tex и synopsis_intro_formatted.tex и там к ним добавляется форматирование. Для научного доклада используется тот же introduction.tex, что и для диссертации, а для автореферата введение пришлось ужать, выкинув большинство ссылок на литературу.

Основной текст диссертации содержится в файлах text/sec-<section_name>.tex и text/intro-<chapter_name>.tex, которые инпутятся в файлы text/chapter-<chapter_name>.tex. Основной текст автореферата и научного доклада - в text/synopsis_main.tex и text/report_main.tex.

Заключение содержится в text/conclusion.tex, и оно общее для всех трёх бумажек.

Приложения содержатся в файле text/appendices.tex, форматируются в formatted_text/appendices_formatted.tex и используются только в диссертации.

Список литературы генерируется в файлах formatted_text/references.tex (для диссертации и научного доклада) и formatted_text/synopsis_references.tex (для автореферата) из bib-файлов в папке biblio. Чужие работы следует помещать в biblio/othercites.bib, свои - в authorpapers.bib. Шаблон даёт возможность помещать свои работы в ВАКовских журналах и тезисы конференций в отдельные файлы biblio/authorpapersVAK.bib и biblio/authorconferences.bib, но я этим не пользовался. Настройки форматирования библиографии - в biblio/biblatex.tex (и немного в formatted_text/references.tex и formatted_text/synopsis_references.tex). Остальные файлы в папке biblio нужны для тех, кто всё-таки пользуется BibTeX'ом, но с ним не получится сгенерировать 2 списка литературы: с чужими и своими работами. В biblio/bibliopreamble.tex происходит автоматическое переключение между настройками BibTeX'а и BibLaTeX'а в зависимости от того, что из них используется.

Картинки в формате pdf кидаются в папку images, для исходников можно создать папку image_sources.

В папке formatting лежат файлы из шаблона, подключающие нужные пакеты и настраивающие форматирование. Мои личные макросы находятся в formatting/latextuning.sty. Из наиболее полезного - вставка картинок командами

\begin{fig}{<label>}{<filename>}
<caption>
\end{fig}

(для широких картинок) и

\begin{narrowfig}{<label>}{<filename>}
<caption>
\end{narrowfig}

(для узких картинок), а также

\begin{eq}{<label>}
<equation with alignment tabs>
\end{eq}

для уравнений с выравниванием.