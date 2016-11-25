# AutomaticQA
select p1.Id, p1.Title, p1.Body, p1.Tags, p1.AcceptedAnswerId, p2.AnswerId, p2.AnswerBody from Posts p1
inner join (
select Id as AnswerId, Body as AnswerBody from Posts 
where Id in (
  select AcceptedAnswerId from Posts
  where Tags like '%<python>%' and AcceptedAnswerId is Not null)) p2
on p1.AcceptedAnswerId = p2.AnswerId
