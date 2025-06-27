tom7-misc
---------

This is a convenience view into https://sourceforge.net/p/tom7misc/svn/ where I
can keep track of a branch with local git operations.

Some docker operations are in [Makefile](Makefile).

To pull more changes from upstream, I do
```shell
git svn fetch
git rebase remotes/git-svn
```
