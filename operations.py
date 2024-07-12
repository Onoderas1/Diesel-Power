import heapq
import itertools
import string
import re
from abc import abstractmethod, ABC
import typing as tp

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for i in rows:
            yield from self.mapper(i)


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for key, group in itertools.groupby(rows, lambda x: [x[k] for k in self.keys]):
            yield from self.reducer(tuple(self.keys), group)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        first = itertools.groupby(rows, lambda x: [x[k] for k in self.keys])
        second = itertools.groupby(args[0], lambda x: [x[k] for k in self.keys])
        heap: list[tp.Any] = []
        last_ind = None
        is_empty1 = False
        is_empty2 = False
        while True:
            if len(heap) == 0:
                try:
                    k1, g1 = first.__next__()
                    heapq.heappush(heap, (k1, 1, g1))
                except StopIteration:
                    is_empty1 = True
                try:
                    k2, g2 = second.__next__()
                    heapq.heappush(heap, (k2, 2, g2))
                except StopIteration:
                    is_empty2 = True
            else:
                if last_ind == 1:
                    try:
                        k1, g1 = first.__next__()
                        heapq.heappush(heap, (k1, 1, g1))
                    except StopIteration:
                        is_empty1 = True
                        k2, g2 = second.__next__()
                        heapq.heappush(heap, (k2, 2, g2))
                else:
                    try:
                        k2, g2 = second.__next__()
                        heapq.heappush(heap, (k2, 2, g2))
                    except StopIteration:
                        is_empty2 = True
                        k1, g1 = first.__next__()
                        heapq.heappush(heap, (k1, 1, g1))
            if is_empty1 and is_empty2:
                break
            last_k, last_ind, last_group = heapq.heappop(heap)
            heapq.heappush(heap, (last_k, last_ind, last_group))
            left: TRowsIterable = []
            right: TRowsIterable = []
            while len(heap) > 0:
                ans_k, ans_ind, ans_group = heapq.heappop(heap)
                if ans_k != last_k:
                    heapq.heappush(heap, (ans_k, ans_ind, ans_group))
                    yield from self.joiner(tuple(self.keys), left, right)
                    break
                else:
                    if ans_ind == 1:
                        left = ans_group
                    else:
                        right = ans_group
                last_k = ans_k
                last_ind = ans_ind
            if len(heap) == 0:
                yield from self.joiner(tuple(self.keys), left, right)


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = row[self.column].translate(str.maketrans('', '', string.punctuation))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self._column = column
        self._separator = separator

    def __call__(self, row: TRow) -> TRowsGenerator:
        pattern = r'[^\s]+' if self._separator is None else fr'[^{self._separator}]*'
        check = True
        for i in re.finditer(pattern, row[self._column]):
            if i[0] != '':
                check = False
                ans = row.copy()
                ans[self._column] = i[0]
                yield ans
        if check:
            ans = row.copy()
            ans[self._column] = ''
            yield ans


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        count = 1
        for word in self.columns:
            count *= row[word]
        row[self.result_column] = count
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        ans = {}
        for key in self.columns:
            ans[key] = row[key]
        yield ans


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        ans: list[tp.Any] = []
        count = 0
        for row in rows:
            count += 1
            heapq.heappush(ans, (row[self.column_max], count, row))
            if len(ans) > self.n:
                heapq.heappop(ans)
        for el in ans:
            yield el[-1]


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        add_word = set()
        count = 0
        ans = []
        for row in rows:
            count += 1
            if row[self.words_column] not in add_word:
                add_word.add(row[self.words_column])
                ans.append(row)
        for v in ans:
            if 'count' not in v:
                v[self.result_column] = 1
                yield v
                break
            v[self.result_column] = v['count'] / count
            del v['count']
            yield v


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        count = 0
        last_row: dict[str, tp.Any] = dict()
        for row in rows:
            count += 1
            last_row = row
        ans = {self.column: count}
        for k in group_key:
            ans[k] = last_row[k]
        yield ans


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        count = 0
        last_row: dict[str, tp.Any] = dict()
        for row in rows:
            count += row[self.column]
            last_row = row
        ans = {self.column: count}
        for k in group_key:
            ans[k] = last_row[k]
        yield ans


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_a)
        if len(rows) > 0:
            for row1 in rows_b:
                ans = row1.copy()
                for row2 in rows:
                    for key in row2:
                        if key not in keys:
                            if key in ans:
                                ans[key + self._b_suffix] = ans[key]
                                del ans[key]
                                ans[key + self._a_suffix] = row2[key]
                            else:
                                ans[key] = row2[key]
                    yield ans
                    ans = row1


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_a)
        is_empty = True
        if len(rows) > 0:
            for row1 in rows_b:
                is_empty = False
                ans = row1.copy()
                for row2 in rows:
                    for key in row2:
                        if key not in keys:
                            if key in ans:
                                ans[key + self._b_suffix] = ans[key]
                                del ans[key]
                                ans[key + self._a_suffix] = row2[key]
                            else:
                                ans[key] = row2[key]
                    yield ans
                    ans = row1
        if len(rows) == 0:
            yield from rows_b
        elif is_empty and (len(rows) > 0):
            yield from rows


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_a)
        is_empty = True
        if len(rows) > 0:
            for row1 in rows_b:
                is_empty = False
                ans = row1.copy()
                for row2 in rows:
                    for key in row2:
                        if key not in keys:
                            if key in ans:
                                ans[key + self._b_suffix] = ans[key]
                                del ans[key]
                                ans[key + self._a_suffix] = row2[key]
                            else:
                                ans[key] = row2[key]
                    yield ans
                    ans = row1
        if is_empty and (len(rows) > 0):
            yield from rows


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        rows = list(rows_a)
        if len(rows) > 0:
            for row1 in rows_b:
                ans = row1.copy()
                for row2 in rows:
                    for key in row2:
                        if key not in keys:
                            if key in ans:
                                ans[key + self._b_suffix] = ans[key]
                                del ans[key]
                                ans[key + self._a_suffix] = row2[key]
                            else:
                                ans[key] = row2[key]
                    yield ans
                    ans = row1
        if len(rows) == 0:
            yield from rows_b
