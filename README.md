Реализация 3-х генераторов: Map, Reduce, Join, которые работают со строками таблицы. Так их будет проще применять - можно будет избегать лишних копирований и задействовать меньше памяти.

1. **`Map`** — операция, которая вызывает переданный генератор (называемый Mapper'ом) от каждой
из строк таблицы. Значения, выданные генератором, образуют таблицу-результат.
(Подходит для элементарных операций над строками - фильтраций, преобразований типов, элементарных
операций над полями таблицы etc).

2. **`Reduce`** принимает на вход таблицу, группирует её строки по ключу (где ключ - значение какого-то
подмножества колонок таблицы) и вызывает Reducer для строк с одинаковым ключом.

3. **`Join`** — самая непростая операция, объединяющая информацию из двух таблиц в одну по ключу.
Строки новой таблицы будут созданы из строк двух таблиц, участвовавших в джойне.

Так же для каждого генератора реализовано несколько функций, которые можно к ним применять для обработки строк таблицы
