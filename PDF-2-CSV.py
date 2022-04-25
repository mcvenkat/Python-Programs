import fitz
import pandas as pd
doc = fitz.open('<Your File Name>.pdf')
page1 = doc[0]
words = page1.get_text("words")

#Extract the coordinates of the first object :
first_annots=[]
rec=page1.first_annot.rect

rec

#Information of words in first object is stored in mywords
mywords = [w for w in words if fitz.Rect(w[:4]) in rec]
ann= make_text(mywords)
first_annots.append(ann)

#This function selects the words contained in the box, sort the words and return in form of a string :

def make_text(words):
    line_dict = {}
    words.sort(key=lambda w: w[0])

    for w in words:
        y1 = round(w[3], 1)
        word = w[4]
        line = line_dict.get(y1, [])
        line.append(word)

Extracting each page of the document and all the annots/rectangles :
  for pageno in range(0,len(doc)-1):
      page = doc[pageno]
      words = page.get_text("words")

      for annot in page.annots():
        if annot!=None:
          rec=annot.rect
          mywords = [w for w in words if fitz.Rect(w[:4]) in rec]
          ann= make_text(mywords)
          all_annots.append(ann)

cont=[]
for i in range(0,len(all_annots)):
    cont.append(all_annots[i].split('n',1))

#Remove unwanted symbols
liss=[]

for i in range(0,len(cont)):
    lis=[]
    for j in cont[i]:
        j=j.replace('*','')
        j=j.replace('#','')
        j=j.replace(':','')
        j=j.strip()
        #print(j)
        lis.append(j)
    liss.append(lis)

Spliting into keys and values and removing spaces in the values which only contain digits :
keys=[]
values=[]
for i in liss:
    keys.append(i[0])
    values.append(i[1])
    for i in range(0, len(values)):
        for j in range(0,len(values[i])):
            if values[i][j]>='A' and values[i][j]<='Z':
            break

        if j==len(values[i])-1:
           values[i]=values[i].replace(' ','')


report=dict(zip(keys,values))

Converting to DataFrame  and exporting to CSV:

data=pd.DataFrame.from_dict(report)
data.to_csv('final.csv',index=False)
