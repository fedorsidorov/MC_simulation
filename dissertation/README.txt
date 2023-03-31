������������� ���� ����� dissertation.tex (��� �����������), synopsis.tex (��� ������������), report.tex (��� �������� �������).

��� ������������ ������������ BibLaTeX, � �� BibTeX, ������� ����� ������������� ��������������� ��������� TeX-���������. � ����������� TeXworks ��� Windows � ������� ���������� ������ "pdfLaTeX+MakeIndex+BibLaTeX", ����������� ������ biblatexify.bat � ����������� $fullname � $basename.

�������� ����� ���������� � ����� text, ��� ��� �� �����������. ���������� ��� ��������� ������ ����������� � text/title_data.tex, ���� ��������� ����� ������������ � formatted_text/dissertation_title.tex, formatted_text/synopsis_title.tex � formatted_text/report_title.tex. ��� �������� ������� ��������� ���������� ��������� ������ � report_title.tex � �� ���������� � title_data.tex.

�������� ���������� � ������ text/introduction.tex � text/synopsis_intro.tex, ������� ����� ��������� � formatted_text/intro_formatted.tex � synopsis_intro_formatted.tex � ��� � ��� ����������� ��������������. ��� �������� ������� ������������ ��� �� introduction.tex, ��� � ��� �����������, � ��� ������������ �������� �������� �����, ������� ����������� ������ �� ����������.

�������� ����� ����������� ���������� � ������ text/sec-<section_name>.tex � text/intro-<chapter_name>.tex, ������� ��������� � ����� text/chapter-<chapter_name>.tex. �������� ����� ������������ � �������� ������� - � text/synopsis_main.tex � text/report_main.tex.

���������� ���������� � text/conclusion.tex, � ��� ����� ��� ���� ��� �������.

���������� ���������� � ����� text/appendices.tex, ������������� � formatted_text/appendices_formatted.tex � ������������ ������ � �����������.

������ ���������� ������������ � ������ formatted_text/references.tex (��� ����������� � �������� �������) � formatted_text/synopsis_references.tex (��� ������������) �� bib-������ � ����� biblio. ����� ������ ������� �������� � biblio/othercites.bib, ���� - � authorpapers.bib. ������ ��� ����������� �������� ���� ������ � ��������� �������� � ������ ����������� � ��������� ����� biblio/authorpapersVAK.bib � biblio/authorconferences.bib, �� � ���� �� �����������. ��������� �������������� ������������ - � biblio/biblatex.tex (� ������� � formatted_text/references.tex � formatted_text/synopsis_references.tex). ��������� ����� � ����� biblio ����� ��� ���, ��� ��-���� ���������� BibTeX'��, �� � ��� �� ��������� ������������� 2 ������ ����������: � ������ � ������ ��������. � biblio/bibliopreamble.tex ���������� �������������� ������������ ����� ����������� BibTeX'� � BibLaTeX'� � ����������� �� ����, ��� �� ��� ������������.

�������� � ������� pdf �������� � ����� images, ��� ���������� ����� ������� ����� image_sources.

� ����� formatting ����� ����� �� �������, ������������ ������ ������ � ������������� ��������������. ��� ������ ������� ��������� � formatting/latextuning.sty. �� �������� ��������� - ������� �������� ���������

\begin{fig}{<label>}{<filename>}
<caption>
\end{fig}

(��� ������� ��������) �

\begin{narrowfig}{<label>}{<filename>}
<caption>
\end{narrowfig}

(��� ����� ��������), � �����

\begin{eq}{<label>}
<equation with alignment tabs>
\end{eq}

��� ��������� � �������������.