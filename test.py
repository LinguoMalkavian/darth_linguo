import tree_handler

testString="""(ROOT
  (sentence
    (sn
      (grup.nom (pd000000 Esta)))
    (grup.verb (vsip000 es))
    (sn
      (spec (da0000 la))
      (grup.nom
        (s.a
          (grup.a (ao0000 primera)))
        (nc0s000 oración)
        (sp
          (prep (sp000 de))
          (sn
            (spec (da0000 el))
            (grup.nom (nc0s000 ejemplo))))))
    (fp .)))"""
sampleTree= tree_handler.loadTree(testString)

print(sampleTree.consolidateText())
